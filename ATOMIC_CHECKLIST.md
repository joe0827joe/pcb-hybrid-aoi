# 🛠️ Atomic Commit Checklist (Quality Gates)

## Phase 1: Code Isolation (Verified during Development)
- [ ] **Single Responsibility**: Does this change address only a single feature, bugfix, or optimization?
- [ ] **No Contamination**: Have all `print()`, `pdb.set_trace()`, and temporary test paths been removed?

## Phase 2: Performance & Specs (Verified during Benchmarking)
- [ ] **Latency Compliance**: Does `scripts/benchmark_performance.py` show stable latency within the **130ms** threshold?
- [ ] **Schema Alignment**: Is the output JSON/Dict format fully compliant with `detection_schema.md`?

## Phase 3: Engineering Quality (Mandatory Testing Standards)
- [ ] **Unit Tests (FIRST)**:
  - **F**ast: Do unit tests execute in under 1 second?
  - **I**ndependent: Are external dependencies removed (e.g., using mocks/fixtures instead of real data folders)?
  - **R**epeatable: Are results consistent across multiple runs and different environments?
  - **S**elf-Validating: Are there automated assertions instead of manual log inspections?
- [ ] **Integration Tests (AABB)**:
  - **A**tomic: Is each test focused specifically on module interface boundaries?
  - **A**mbient-Independent: **Strictly no hardcoded absolute paths**; use relative paths or `Path(__file__)`.
  - **B**oundary-Focused: Are test cases concentrated on data exchange between modules?
  - **B**ehavior-Driven: Do test cases clearly describe the expected system behavior?

## Phase 4: DevOps & Git Protocol (Pre-Commit Verification)
- [ ] **Environment Sync**: If new libraries were added, has `requirements.txt` been updated?
- [ ] **Commit Compliance**: Does the commit message follow the `type(scope): subject` (Conventional Commits) format?
- [ ] **Documentation Sync**: Has the Project Tree in `README.md` been updated to reflect these changes?
