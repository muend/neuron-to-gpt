# YZ50 submission materials

The Week 11 and Week 12 submission is shared through GitHub links. No email
has been sent by the agent. Local export ZIPs are ignored and are not uploaded
or required as email attachments.

- [Short Turkish email draft](teslim_maili.txt)
- [Email summary image](yz50_hafta11_12_sonuclar.png)
- [Week 11 code and instructions](../week11/README.md)
- [Week 11 measured GPU report](../week11/results/nvidia_l4/submission_report.md)
- [Week 12 code and instructions](../week12/README.md)
- [Week 12 measured GPU report](../week12/results/nvidia_l4/submission_report.md)

The figure is generated from the verified JSON evidence with
`python deliverables/make_submission_figure.py`. It shows cumulative speed
measurements, held-out loss, and both models' matching HellaSwag evaluation.

The Week 12 returned evidence was imported with `import_week12.py`; checks are
recorded in `../week12/results/nvidia_l4/import_validation.json`. Import checks
archived source hashes, finite metrics, token/update accounting and prediction
argmins/counts; it does not rerun GPU inference.

The user-enabled Google Drive export separately preserves the token archive
and resumable `last.pt` under `MyDrive/YZ50/week12/nvidia_l4/`. The large token
and checkpoint files are excluded from Git. Expected checkpoint SHA-256:
`3151b14452b1a62ef7e0722acdd3071025700da2881c3950994536e3aedc97f3`.

A later G4 experiment uses a separate output directory and retains the completed
L4 assignment evidence.
