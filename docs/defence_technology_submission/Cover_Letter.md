# Cover Letter — Defence Technology

Dear Editor-in-Chief,

We are pleased to submit our manuscript entitled **"A Dual-System Enhanced Large Language Model Architecture for Real-Time Joint-Operation Mission Planning in Command and Control Systems"** for consideration in *Defence Technology*.

This work lies at the interdisciplinary frontier of artificial intelligence and military command and control (C2). It addresses a question that is increasingly pressing for modern multi-domain operations: *how can a large language model (LLM) be made to behave as a credible, real-time-capable planning engine for joint-operation courses of action, rather than as an uncontrolled text generator?* We believe the work is a strong fit for *Defence Technology*'s scope on LLM applications in C2, wargaming, operational planning, and kill-chain optimization.

We highlight three contributions that we see as the core defence value of the paper:

1. **Evidence-grounded cognitive architecture.** We present the first (to our knowledge) tightly coupled planning loop combining case-based retrieval (Memento), adaptive stochastic capability control (HOPE), and reflection-driven prompt repair (Reflection). The three mechanisms share a common capability representation and are evaluated end-to-end on a single traceable pipeline, rather than described in isolation. On a reference scenario, the full loop improves mission success from 62.75 to 65.76 over a pure-LLM baseline, and a five-seed paired replication confirms the gain is statistically significant (t(4)=3.75, p=0.02, Cohen's d=1.68); across five scene-out scenarios it improves the pure baseline in four of five families.

2. **Dual-system real-time execution.** We resolve the high-latency bottleneck that makes LLM planning impractical in live C2 by introducing a decoupled dual-system path: a rule-based emergency tier that emits a structurally complete plan and its DoDAF/C2SIM export in 5.6 ms (1.4 ms warm) with no LLM call, and a bottleneck-driven delta-patching tier that refines that anchor to 96.4% of the full-loop quality within 22.8 s, versus 365 s for the complete loop. This yields a three-tier latency profile — millisecond emergency plan, tens-of-seconds agile tactical refinement, and minutes-level full optimization — and we demonstrate empirically that the delta-patching mechanism is monotone and preserves the high-quality anchor that full regeneration destroys.

3. **Standard-compliant interoperability.** The system natively emits DoDAF OV-5b/OV-6c architecture views and SISO C2SIM (IEEE 1516) interoperation artifacts, making the generated courses of action directly consumable by existing C2 terminals and distributed joint-simulation federations. It is engine-agnostic: it does not compete with physics-based simulators such as OneSAF or FLAMES, but serves as the standard planning hub that bridges LLM output into those environments.

We believe the manuscript is well suited to the readership of *Defence Technology*: it combines a concrete C2 problem, a principled systems-design answer, and reproducible quantitative evidence. All scenario configurations, evaluation logs, and software artifacts are available from the corresponding author upon reasonable request.

We respectfully suggest reviewers with expertise in the following areas, should they be available: command-and-control decision-making (C2 decision-making), military LLM agents, and C2-simulation interoperability (C2SIM/DoDAF).

Thank you for your time and consideration. We look forward to your review.

Sincerely,
Changrui Zhang
Chengwei Yang (corresponding author, yangchengwei@bit.edu.cn)
Beijing Institute of Technology
