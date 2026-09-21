# Main branch working rules

- Main is the submission branch: keep one root README, the technical design, product code, tests and necessary documentation.
- Do not restore stage plans, development diaries, duplicate setup guides or archived reviews into main.
- The persistent problem log and development history remain in the dev branch as `actual problems.md`; preserve its IDs and records there.
- Published EXE, PPTX and DOCX belong in GitHub Releases. Do not duplicate them in main.
- Preserve the measured_at/available_at contract, fail-closed checks and supports_actions=false.
- Verify relevant behavior and documentation links before publishing. Existing releases and their tags are immutable; publish a new version if delivered files change.
