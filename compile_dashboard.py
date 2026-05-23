#!/usr/bin/env python3
"""
compile_dashboard.py - Combines all clinical audit runs from reasoning_log.jsonl 
and disagreement_log.jsonl to produce a premium, unified visual HTML Dashboard 
and a consolidated Markdown report.
"""

if __name__ == "__main__":
    import os
    import sys
    import subprocess
    from pathlib import Path

    venv_dir = Path(__file__).parent.resolve() / "venv"
    if venv_dir.exists():
        try:
            is_in_venv = Path(sys.executable).resolve().is_relative_to(venv_dir)
        except AttributeError:
            try:
                Path(sys.executable).resolve().relative_to(venv_dir)
                is_in_venv = True
            except ValueError:
                is_in_venv = False

        if not is_in_venv:
            if os.name == "nt":
                venv_python = venv_dir / "Scripts" / "python.exe"
            else:
                venv_python = venv_dir / "bin" / "python"

            if venv_python.exists():
                args = [str(venv_python)] + sys.argv
                sys.exit(subprocess.call(args))

import json
import os
import re
from datetime import datetime
from pathlib import Path

# Paths
REPORTS_DIR = Path(__file__).parent / "reports"
LOGS_DIR = Path(__file__).parent / "logs"
REASONING_LOG = LOGS_DIR / "reasoning_log.jsonl"
DISAGREEMENT_LOG = LOGS_DIR / "disagreement_log.jsonl"

def safe_parse_jsonl(file_path: Path) -> list[dict]:
    """Parse JSONL files safely, handling NaN values and malformed lines."""
    records = []
    if not file_path.exists():
        return records
    
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            # Replace invalid NaN with null for robust parsing
            cleaned_line = re.sub(r"\bNaN\b", "null", line)
            try:
                data = json.loads(cleaned_line)
                records.append(data)
            except json.JSONDecodeError as exc:
                print(f"[WARN] Failed to parse line {i} in {file_path.name}: {exc}")
    return records

def build_consolidated_data() -> tuple[list[dict], dict]:
    """Merge, deduplicate (keeping the latest run per patient), and compile stats."""
    consensus_runs = safe_parse_jsonl(REASONING_LOG)
    disagreement_runs = safe_parse_jsonl(DISAGREEMENT_LOG)
    
    # Track the latest run per patient
    patient_latest: dict[str, dict] = {}
    
    # Process successful consensus runs
    for run in consensus_runs:
        pid = run.get("patient_id")
        if not pid:
            continue
        
        # Parse timestamp
        ts_str = run.get("timestamp", "")
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except ValueError:
            ts = datetime.min
            
        kappa = run.get("kappa_score")
        # In case it is null or string
        try:
            kappa_val = float(kappa) if kappa is not None else None
        except ValueError:
            kappa_val = None
            
        record = {
            "patient_id": pid,
            "status": "CONSENSUS",
            "kappa": kappa_val,
            "iterations": run.get("iterations", 1),
            "timestamp": ts,
            "timestamp_str": ts_str,
            "icd_codes": run.get("final_icd_codes", []),
            "grounding": run.get("grounding_table", []),
            "diagnosis": "Consensus Reached (See patient details)",
            "evidence": []
        }
        
        # Check if we have a newer run already
        if pid not in patient_latest or ts > patient_latest[pid]["timestamp"]:
            patient_latest[pid] = record

    # Process disagreement runs
    for run in disagreement_runs:
        pid = run.get("patient_id")
        if not pid:
            continue
            
        ts_str = run.get("timestamp", "")
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except ValueError:
            ts = datetime.min
            
        kappa = run.get("final_kappa")
        try:
            kappa_val = float(kappa) if kappa is not None else None
        except ValueError:
            kappa_val = None
            
        diag_hyp = ""
        evidence = []
        diag_final = run.get("diagnostician_final_position")
        if diag_final and isinstance(diag_final, dict):
            diag_hyp = diag_final.get("diagnosis_hypothesis", "")
            evidence = diag_final.get("evidence_chain", [])
            
        record = {
            "patient_id": pid,
            "status": "ESCALATED",
            "kappa": kappa_val,
            "iterations": run.get("iterations", 3),
            "timestamp": ts,
            "timestamp_str": ts_str,
            "icd_codes": diag_final.get("icd_codes_cited", []) if diag_final else [],
            "grounding": [],
            "diagnosis": diag_hyp or "Requires Physician Review",
            "evidence": evidence
        }
        
        if pid not in patient_latest or ts > patient_latest[pid]["timestamp"]:
            patient_latest[pid] = record

    # Sort patient list by patient ID
    sorted_patients = [patient_latest[pid] for pid in sorted(patient_latest.keys())]
    
    # Calculate global stats
    total = len(sorted_patients)
    consensus_count = sum(1 for p in sorted_patients if p["status"] == "CONSENSUS")
    escalated_count = sum(1 for p in sorted_patients if p["status"] == "ESCALATED")
    
    valid_kappas = [p["kappa"] for p in sorted_patients if p["kappa"] is not None]
    avg_kappa = sum(valid_kappas) / len(valid_kappas) if valid_kappas else 0.0
    
    stats = {
        "total_audited": total,
        "consensus_count": consensus_count,
        "escalated_count": escalated_count,
        "consensus_rate_pct": round((consensus_count / total * 100), 1) if total > 0 else 0.0,
        "avg_kappa": round(avg_kappa, 3)
    }
    
    return sorted_patients, stats

def generate_markdown_report(patients: list[dict], stats: dict) -> Path:
    """Write a consolidated Markdown report to reports/consolidated_report.md."""
    REPORTS_DIR.mkdir(exist_ok=True)
    report_path = REPORTS_DIR / "consolidated_report.md"
    
    lines = [
        "# Consolidated Clinical Audit Report",
        "",
        f"**Compiled at**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Total Patients Audited**: {stats['total_audited']}  ",
        f"**Consensus Rate**: {stats['consensus_rate_pct']}% ({stats['consensus_count']}/{stats['total_audited']})  ",
        f"**Average κ Agreement Score**: {stats['avg_kappa']:.3f}  ",
        "",
        "## Summary Metrics",
        "",
        f"- **Consensus Reached**: {stats['consensus_count']}",
        f"- **Escalated for Physician Review**: {stats['escalated_count']}",
        f"- **Average Agreement Score (κ)**: {stats['avg_kappa']:.3f}",
        "",
        "## Patient Audit Status Table",
        "",
        "| Patient ID | Audit Status | Agreement (κ) | Iterations | Grounded Codes | Last Audited Time |",
        "|:-----------|:-------------|--------------:|-----------:|:---------------|:------------------|"
    ]
    
    for p in patients:
        kappa_s = f"{p['kappa']:.3f}" if p["kappa"] is not None else "—"
        iters_s = str(p["iterations"])
        codes_s = ", ".join(p["icd_codes"][:8]) + ("..." if len(p["icd_codes"]) > 8 else "")
        ts_clean = p["timestamp_str"].split(".")[0].replace("T", " ") if p["timestamp_str"] else "—"
        
        status_label = "🟢 CONSENSUS" if p["status"] == "CONSENSUS" else "🔴 ESCALATED"
        lines.append(
            f"| **{p['patient_id']}** | {status_label} | {kappa_s} | {iters_s} | `{codes_s or 'None'}` | {ts_clean} |"
        )
        
    lines += [
        "",
        "---",
        "*Note: This report is dynamically compiled from reasoning logs (`reasoning_log.jsonl` and `disagreement_log.jsonl`). It automatically aggregates multiple sessions and de-duplicates patients to present the absolute latest status for each clinical audit.*"
    ]
    
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path

def generate_html_dashboard(patients: list[dict], stats: dict) -> Path:
    """Create a premium, glassmorphism dark-mode HTML dashboard."""
    REPORTS_DIR.mkdir(exist_ok=True)
    dashboard_path = REPORTS_DIR / "dashboard.html"
    
    # Convert patient datetime to string for JSON serialization
    serialized_patients = []
    for p in patients:
        p_copy = p.copy()
        p_copy["timestamp"] = p["timestamp"].isoformat() if p["timestamp"] != datetime.min else ""
        serialized_patients.append(p_copy)
        
    patients_json = json.dumps(serialized_patients, indent=2)
    
    html_content = f"""<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Clinical Audit Dashboard</title>
    <!-- Outfit Font & Tailwind CSS -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {{
            darkMode: 'class',
            theme: {{
                extend: {{
                    fontFamily: {{
                        sans: ['Outfit', 'sans-serif'],
                    }},
                }}
            }}
        }}
    </script>
    <style>
        body {{
            background: radial-gradient(circle at 50% 0%, #1a1e36 0%, #0d0f1a 100%);
            min-height: 100vh;
        }}
        .glass {{
            background: rgba(30, 41, 59, 0.45);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.08);
        }}
        .glass-card {{
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }}
        .glass-card:hover {{
            transform: translateY(-4px);
            border-color: rgba(99, 102, 241, 0.3);
            box-shadow: 0 12px 30px -10px rgba(99, 102, 241, 0.2);
        }}
        .custom-scrollbar::-webkit-scrollbar {{
            width: 6px;
            height: 6px;
        }}
        .custom-scrollbar::-webkit-scrollbar-track {{
            background: rgba(0, 0, 0, 0.1);
        }}
        .custom-scrollbar::-webkit-scrollbar-thumb {{
            background: rgba(255, 255, 255, 0.15);
            border-radius: 9999px;
        }}
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {{
            background: rgba(255, 255, 255, 0.3);
        }}
    </style>
</head>
<body class="text-slate-100 font-sans custom-scrollbar antialiased p-6 sm:p-10">

    <div class="max-w-7xl mx-auto space-y-10">
        
        <!-- Header -->
        <header class="flex flex-col md:flex-row md:items-center md:justify-between gap-4 pb-6 border-b border-slate-800">
            <div>
                <div class="flex items-center gap-3">
                    <span class="px-3 py-1 text-xs font-semibold tracking-wider text-indigo-400 bg-indigo-500/10 rounded-full uppercase">Operations Room</span>
                </div>
                <h1 class="text-3xl sm:text-4xl font-bold tracking-tight text-white mt-2">Multi-Agent Clinical Auditor</h1>
                <p class="text-slate-400 mt-1">Unified consensus auditing dashboard across longitudinal EHR patient cohorts</p>
            </div>
            <div class="text-right">
                <span class="text-xs text-slate-500 block">LAST REFRESHED</span>
                <span class="text-sm font-medium text-slate-300">{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</span>
            </div>
        </header>

        <!-- Stats Cards Grid -->
        <section class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
            
            <!-- Total Audited -->
            <div class="glass glass-card p-6 rounded-2xl flex flex-col justify-between">
                <span class="text-sm font-medium text-slate-400 uppercase tracking-wider">Total Cohort Audited</span>
                <div class="flex items-baseline gap-2 mt-4">
                    <span class="text-4xl font-bold text-white">{stats['total_audited']}</span>
                    <span class="text-slate-500 text-sm">Patients</span>
                </div>
                <div class="h-1.5 w-full bg-slate-800 rounded-full mt-4 overflow-hidden">
                    <div class="h-full bg-blue-500 rounded-full" style="width: 100%"></div>
                </div>
            </div>

            <!-- Consensus Rate -->
            <div class="glass glass-card p-6 rounded-2xl flex flex-col justify-between">
                <span class="text-sm font-medium text-slate-400 uppercase tracking-wider">Consensus Accuracy</span>
                <div class="flex items-baseline gap-2 mt-4">
                    <span class="text-4xl font-bold text-emerald-400">{stats['consensus_rate_pct']}%</span>
                    <span class="text-slate-500 text-sm">{stats['consensus_count']}/{stats['total_audited']} reached</span>
                </div>
                <div class="h-1.5 w-full bg-slate-800 rounded-full mt-4 overflow-hidden">
                    <div class="h-full bg-emerald-400 rounded-full" style="width: {stats['consensus_rate_pct']}%"></div>
                </div>
            </div>

            <!-- Escalated count -->
            <div class="glass glass-card p-6 rounded-2xl flex flex-col justify-between">
                <span class="text-sm font-medium text-slate-400 uppercase tracking-wider">Physician Escalations</span>
                <div class="flex items-baseline gap-2 mt-4">
                    <span class="text-4xl font-bold text-amber-500">{stats['escalated_count']}</span>
                    <span class="text-slate-500 text-sm">Review required</span>
                </div>
                <div class="h-1.5 w-full bg-slate-800 rounded-full mt-4 overflow-hidden">
                    <div class="h-full bg-amber-500 rounded-full" style="width: {100 - stats['consensus_rate_pct']}%"></div>
                </div>
            </div>

            <!-- Avg Kappa -->
            <div class="glass glass-card p-6 rounded-2xl flex flex-col justify-between">
                <span class="text-sm font-medium text-slate-400 uppercase tracking-wider">Mean Agreement Score</span>
                <div class="flex items-baseline gap-2 mt-4">
                    <span class="text-4xl font-bold text-indigo-400">κ={stats['avg_kappa']:.3f}</span>
                </div>
                <div class="h-1.5 w-full bg-slate-800 rounded-full mt-4 overflow-hidden">
                    <div class="h-full bg-indigo-400 rounded-full" style="width: {min(100, int(stats['avg_kappa'] * 100)) if stats['avg_kappa'] > 0 else 0}%"></div>
                </div>
            </div>

        </section>

        <!-- Main Content Table and Search -->
        <main class="glass rounded-2xl overflow-hidden">
            
            <!-- Controls Bar -->
            <div class="p-6 border-b border-slate-800 bg-slate-900/40 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
                <div class="relative max-w-md w-full">
                    <div class="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                        <svg class="h-5 w-5 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path></svg>
                    </div>
                    <input type="text" id="searchInput" oninput="filterPatients()" placeholder="Search by Patient ID, codes, or diagnosis..." class="w-full bg-slate-950/60 border border-slate-800 text-slate-200 pl-10 pr-4 py-2 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent text-sm">
                </div>
                
                <div class="flex items-center gap-3">
                    <button id="btn-all" onclick="setStatusFilter('ALL')" class="px-4 py-1.5 rounded-lg text-xs font-semibold transition bg-indigo-600 text-white shadow-lg">All Statuses</button>
                    <button id="btn-consensus" onclick="setStatusFilter('CONSENSUS')" class="px-4 py-1.5 rounded-lg text-xs font-semibold text-slate-400 hover:bg-slate-800 transition">Consensus</button>
                    <button id="btn-escalated" onclick="setStatusFilter('ESCALATED')" class="px-4 py-1.5 rounded-lg text-xs font-semibold text-slate-400 hover:bg-slate-800 transition">Escalated</button>
                </div>
            </div>

            <!-- Table Wrapper -->
            <div class="overflow-x-auto custom-scrollbar">
                <table class="w-full text-left border-collapse">
                    <thead>
                        <tr class="border-b border-slate-800 text-slate-400 text-xs font-semibold uppercase tracking-wider bg-slate-900/10">
                            <th class="p-6">Patient ID</th>
                            <th class="p-6">Audit Status</th>
                            <th class="p-6">Agreement (κ)</th>
                            <th class="p-6">Iterations</th>
                            <th class="p-6">Cited ICD Codes</th>
                            <th class="p-6 text-right">Actions</th>
                        </tr>
                    </thead>
                    <tbody id="tableBody" class="divide-y divide-slate-800/60">
                        <!-- Dynamic content loaded via JS -->
                    </tbody>
                </table>
            </div>
            
            <div id="noResults" class="hidden p-12 text-center text-slate-500">
                No audited patients matched your search criteria.
            </div>
            
        </main>
        
        <!-- Detailed Modal -->
        <div id="detailModal" class="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm hidden" onclick="closeModal(event)">
            <div class="glass max-w-3xl w-full max-h-[85vh] rounded-2xl overflow-hidden flex flex-col shadow-2xl" onclick="event.stopPropagation()">
                <!-- Modal Header -->
                <div class="p-6 border-b border-slate-800 bg-slate-900/60 flex items-center justify-between">
                    <div>
                        <h3 id="modalTitle" class="text-xl font-bold text-white">Patient Details</h3>
                        <span id="modalTimestamp" class="text-xs text-slate-500 mt-1 block"></span>
                    </div>
                    <button onclick="hideModal()" class="p-2 hover:bg-slate-800 rounded-lg text-slate-400 hover:text-slate-100 transition">
                        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path></svg>
                    </button>
                </div>
                
                <!-- Modal Body -->
                <div class="p-6 space-y-6 overflow-y-auto custom-scrollbar text-sm" id="modalBody">
                    
                    <!-- Stats summary inside modal -->
                    <div class="grid grid-cols-3 gap-4 p-4 rounded-xl bg-slate-950/40 border border-slate-800">
                        <div>
                            <span class="text-xs text-slate-400 block uppercase font-medium">Audit Status</span>
                            <span id="modalStatusBadge" class="inline-block mt-1 font-semibold"></span>
                        </div>
                        <div>
                            <span class="text-xs text-slate-400 block uppercase font-medium">Kappa Score</span>
                            <span id="modalKappa" class="inline-block mt-1 text-base font-bold"></span>
                        </div>
                        <div>
                            <span class="text-xs text-slate-400 block uppercase font-medium">Conversations</span>
                            <span id="modalIterations" class="inline-block mt-1 text-base font-medium"></span>
                        </div>
                    </div>

                    <!-- Hypothesis -->
                    <div class="space-y-2">
                        <h4 class="font-bold text-white uppercase text-xs tracking-wider text-indigo-400">Diagnosis Summary / Hypothesis</h4>
                        <div id="modalHypothesis" class="p-4 rounded-xl bg-slate-900/40 border border-slate-800 text-slate-300 leading-relaxed italic">
                        </div>
                    </div>

                    <!-- ICD codes -->
                    <div class="space-y-2">
                        <h4 class="font-bold text-white uppercase text-xs tracking-wider text-indigo-400">Audited ICD Codes</h4>
                        <div id="modalCodes" class="flex flex-wrap gap-2">
                        </div>
                    </div>

                    <!-- Evidence / Contradictions / Grounding -->
                    <div id="modalEvidenceSection" class="space-y-2">
                        <h4 class="font-bold text-white uppercase text-xs tracking-wider text-indigo-400">Evidence Chain & Grounding Audit</h4>
                        <div id="modalEvidenceList" class="space-y-2.5">
                        </div>
                    </div>

                </div>
                
                <!-- Modal Footer -->
                <div class="p-6 border-t border-slate-800 bg-slate-900/30 flex justify-end">
                    <button onclick="hideModal()" class="px-5 py-2 bg-slate-800 hover:bg-slate-700 transition rounded-xl text-sm font-semibold">Close Details</button>
                </div>
            </div>
        </div>

    </div>

    <!-- Inject Patient JSON Data -->
    <script>
        const patients = {patients_json};
        
        let currentStatusFilter = 'ALL';
        
        function getKappaLabel(k) {{
            if (k === null || k === undefined) return '—';
            return k.toFixed(3);
        }}
        
        function renderTable(list) {{
            const tbody = document.getElementById('tableBody');
            const noResults = document.getElementById('noResults');
            tbody.innerHTML = '';
            
            if (list.length === 0) {{
                noResults.classList.remove('hidden');
                return;
            }}
            noResults.classList.add('hidden');
            
            list.forEach(p => {{
                const tr = document.createElement('tr');
                tr.className = 'hover:bg-slate-800/20 transition cursor-pointer';
                tr.onclick = () => showModal(p.patient_id);
                
                const badgeClass = p.status === 'CONSENSUS' 
                    ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' 
                    : 'bg-amber-500/10 text-amber-400 border-amber-500/20';
                    
                const badgeText = p.status === 'CONSENSUS' ? '🟢 Consensus' : '🔴 Escalated';
                
                const kappaText = getKappaLabel(p.kappa);
                
                const codeBadges = p.icd_codes.slice(0, 5).map(c => 
                    `<span class="px-2 py-0.5 bg-slate-950/60 rounded text-slate-300 font-mono text-xs border border-slate-800">${{c}}</span>`
                ).join(' ');
                
                const codeSuffix = p.icd_codes.length > 5 ? `<span class="text-xs text-slate-500 font-medium"> +${{p.icd_codes.length - 5}} more</span>` : '';
                
                tr.innerHTML = `
                    <td class="p-6 font-bold text-white">${{p.patient_id}}</td>
                    <td class="p-6">
                        <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border ${{badgeClass}}">
                            ${{badgeText}}
                        </span>
                    </td>
                    <td class="p-6 font-semibold text-slate-200">${{kappaText}}</td>
                    <td class="p-6 text-slate-400">${{p.iterations}} iter</td>
                    <td class="p-6">
                        <div class="flex items-center gap-1.5 flex-wrap max-w-md">
                            ${{codeBadges || '<span class="text-slate-500 italic text-xs">No codes cited</span>'}}
                            ${{codeSuffix}}
                        </div>
                    </td>
                    <td class="p-6 text-right">
                        <button class="px-3.5 py-1.5 bg-indigo-500/10 hover:bg-indigo-600/20 text-indigo-400 hover:text-indigo-300 rounded-lg text-xs font-semibold border border-indigo-500/20 transition">
                            View Audit
                        </button>
                    </td>
                `;
                tbody.appendChild(tr);
            }});
        }}
        
        function setStatusFilter(status) {{
            currentStatusFilter = status;
            
            // Update active state of buttons
            const btns = {{
                'ALL': document.getElementById('btn-all'),
                'CONSENSUS': document.getElementById('btn-consensus'),
                'ESCALATED': document.getElementById('btn-escalated')
            }};
            
            Object.keys(btns).forEach(key => {{
                if (!btns[key]) return;
                if (key === status) {{
                    btns[key].className = "px-4 py-1.5 rounded-lg text-xs font-semibold bg-indigo-600 text-white shadow-lg transition";
                }} else {{
                    btns[key].className = "px-4 py-1.5 rounded-lg text-xs font-semibold text-slate-400 hover:bg-slate-800 transition";
                }}
            }});
            
            filterPatients();
        }}
        
        function filterPatients() {{
            const search = document.getElementById('searchInput').value.toLowerCase().trim() || '';
            
            const filtered = patients.filter(p => {{
                const matchesStatus = currentStatusFilter === 'ALL' || p.status === currentStatusFilter;
                
                const matchesSearch = !search 
                    || p.patient_id.toLowerCase().includes(search)
                    || p.diagnosis.toLowerCase().includes(search)
                    || p.icd_codes.some(c => c.toLowerCase().includes(search));
                    
                return matchesStatus && matchesSearch;
            }});
            
            renderTable(filtered);
        }}
        
        function showModal(patientId) {{
            const p = patients.find(x => x.patient_id === patientId);
            if (!p) return;
            
            document.getElementById('modalTitle').innerText = `Clinical Audit: Patient ${{p.patient_id}}`;
            
            const dateStr = p.timestamp_str ? p.timestamp_str.split('.')[0].replace('T', ' ') + ' UTC' : 'Unknown date';
            document.getElementById('modalTimestamp').innerText = `Audited on: ${{dateStr}}`;
            
            const badge = document.getElementById('modalStatusBadge');
            badge.innerText = p.status === 'CONSENSUS' ? '✓ Consensus' : '⚠ Escalated';
            badge.className = p.status === 'CONSENSUS' 
                ? 'inline-block mt-1 font-semibold text-emerald-400' 
                : 'inline-block mt-1 font-semibold text-amber-500';
                
            document.getElementById('modalKappa').innerText = p.kappa !== null ? p.kappa.toFixed(3) : '—';
            document.getElementById('modalIterations').innerText = `${{p.iterations}} negotiation round(s)`;
            
            // Hypothesis
            const hypothesisBox = document.getElementById('modalHypothesis');
            if (p.status === 'CONSENSUS') {{
                hypothesisBox.innerText = `Audit Consensus reached. All agents agreed on the patient diagnosis and ICD code assignments after ${{p.iterations}} negotiation iterations.`;
            }} else {{
                hypothesisBox.innerText = p.diagnosis || 'Diagnosis discrepancies between agents required escalation to standard clinician overview.';
            }}
            
            // Codes
            const codesBox = document.getElementById('modalCodes');
            codesBox.innerHTML = '';
            if (p.icd_codes.length === 0) {{
                codesBox.innerHTML = '<span class="text-slate-500 italic">No codes recorded</span>';
            }} else {{
                p.icd_codes.forEach(c => {{
                    const span = document.createElement('span');
                    span.className = 'px-3 py-1 bg-slate-950/60 text-slate-300 font-mono text-xs rounded-xl border border-slate-800 shadow-sm';
                    span.innerText = c;
                    codesBox.appendChild(span);
                }});
            }}
            
            // Grounding & Evidence List
            const listEl = document.getElementById('modalEvidenceList');
            listEl.innerHTML = '';
            
            if (p.status === 'CONSENSUS' && p.grounding && p.grounding.length > 0) {{
                p.grounding.forEach(g => {{
                    const item = document.createElement('div');
                    item.className = 'p-3.5 rounded-xl bg-slate-950/30 border border-slate-800 flex items-start justify-between gap-4';
                    
                    const codeLabel = `<span class="px-2 py-0.5 bg-slate-900 rounded font-mono text-xs text-white border border-slate-800">ICD-${{g.icd_version}} ${{g.icd_code}}</span>`;
                    
                    const titleText = g.title ? `<span class="text-slate-300 ml-2 font-medium font-sans">${{g.title}}</span>` : '<span class="text-slate-500 ml-2 italic">Unlabeled medical concept</span>';
                    
                    const stageBadge = g.stage 
                        ? `<span class="px-2 py-0.5 rounded-full text-[10px] font-semibold uppercase tracking-wider bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">Stage ${{g.stage}} Verified</span>`
                        : `<span class="px-2 py-0.5 rounded-full text-[10px] font-semibold uppercase tracking-wider bg-rose-500/10 text-rose-400 border border-rose-500/20">Unverified</span>`;
                        
                    item.innerHTML = `
                        <div class="flex items-center gap-1">
                            ${{codeLabel}}
                            ${{titleText}}
                        </div>
                        <div>
                            ${{stageBadge}}
                        </div>
                    `;
                    listEl.appendChild(item);
                }});
            }} else if (p.evidence && p.evidence.length > 0) {{
                p.evidence.forEach(ev => {{
                    const item = document.createElement('div');
                    item.className = 'p-3.5 rounded-xl bg-slate-950/30 border border-slate-800 flex items-start gap-3';
                    item.innerHTML = `
                        <span class="text-amber-500 shrink-0 select-none">✦</span>
                        <p class="text-slate-300 leading-relaxed font-sans text-xs">${{ev}}</p>
                    `;
                    listEl.appendChild(item);
                }});
            }} else {{
                listEl.innerHTML = '<div class="p-4 bg-slate-950/20 text-slate-500 italic text-center rounded-xl border border-slate-800">No additional logs or evidence records are active.</div>';
            }}
            
            document.getElementById('detailModal').classList.remove('hidden');
        }}
        
        function hideModal() {{
            document.getElementById('detailModal').classList.add('hidden');
        }}
        
        function closeModal(event) {{
            if (event.target === document.getElementById('detailModal')) {{
                hideModal();
            }}
        }}
        
        // Initial table load
        document.addEventListener('DOMContentLoaded', () => {{
            renderTable(patients);
        }});
    </script>
</body>
</html>
"""
    dashboard_path.write_text(html_content, encoding="utf-8")
    return dashboard_path

def main() -> None:
    print("[DASHBOARD] Scanning JSONL logs to compile clinical audits...")
    patients, stats = build_consolidated_data()
    
    if not patients:
        print("[ERROR] No clinical audits found in logs directory. Please run batch_run.py first.")
        return
        
    md_path = generate_markdown_report(patients, stats)
    html_path = generate_html_dashboard(patients, stats)
    
    print("═" * 60)
    print("  CLINICAL AUDIT LOGS CONSOLIDATION COMPLETE".center(60))
    print("═" * 60)
    print(f"  Total Audited Patients : {stats['total_audited']}")
    print(f"  Consensus Reached      : {stats['consensus_count']} ({stats['consensus_rate_pct']}%)")
    print(f"  Physician Escalated    : {stats['escalated_count']}")
    print(f"  Mean Agreement κ Score : {stats['avg_kappa']:.3f}")
    print("─" * 60)
    print(f"  [+] Markdown Report    : {md_path.absolute()}")
    print(f"  [+] HTML Dashboard     : {html_path.absolute()}")
    print("═" * 60)
    print()

if __name__ == "__main__":
    main()
