# Contributing

Thank you for considering contributing to HEAR-Kit!  
We welcome contributions of all kinds — new metrics, dataset and task integrations, bug fixes, documentation improvements, and more.  

Please follow the guidelines below to keep the project consistent and easy to maintain.

---

## Branch Naming

All branches must follow the pattern:
<type>/<short-description>

### Allowed `<type>` prefixes
- `feat` – new feature (e.g., new metric, task, dataset, evaluator)
- `fix` – bug fix
- `docs` – documentation changes
- `test` – tests or fixtures
- `refactor` – refactor code without changing behavior
- `perf` – performance improvements
- `ci` – CI/CD or automation changes
- `build` – build system or dependencies
- `style` – formatting/naming only (no logic change)
- `chore` – maintenance tasks
- `revert` – revert a commit

Special:
- `release/` – release preparation (e.g., `release/v0.1.0`)
- `hotfix/` – urgent fixes

### Short description rules
- Use **kebab-case**: `fix/audio-loader-bug` ✅, `fix/AudioLoaderBug` ❌
- Keep it concise (≤ 5 words) and descriptive of the change
- Do **not** include usernames, spaces, or special characters

### Examples
- `feat/metric-wer-support`
- `fix/dataloader-stereo-mismatch`
- `docs/benchmarking-guide`
- `test/asr-librispeech-suite`
- `release/v0.1.0`

---

## Pull Request Workflow

We use the **fork → branch → PR** model for contributions.  
If you are an external contributor, please work from your fork.  
If you are a maintainer with write access, you may create branches directly on this repository (see Maintainer Notes below).

### 🚀 TL;DR for First-Time Contributors
1. Fork the repo and clone your fork  
2. Create a branch → make changes → commit  
3. Push to your fork and open a Pull Request to `upstream:main`  

---

### Steps for Contributors

1. **Sync your fork with upstream**:  
   ```bash
   git checkout main
   git pull origin main
   git fetch upstream
   git merge upstream/main

   Replace upstream with the remote name pointing to the original repository.

2. **Create a branch following the naming rules**:
   ```bash
   git checkout -b <type>/<short-description>
   ```
   Examples:
   - fix/typo-docs
   - feat/add-audio-eval

3. **Commit changes with clear messages**:
    ```bash
    git commit -m "fix: correct typo in README"
    ```

4. **Push your branch to your fork**:
    ```bash
    git push origin <type>/<short-description>
    ```

5. **Create a pull request**:
    - Go to your fork on GitHub and click "New Pull Request"
    - Compare your branch against upstream:main
    - Fill in the PR template with context, motivation, and test details
    - Mark it as a Draft PR if the work is still in progress  
   
6. **Keep your PR up to date**:
    - Rebase frequently on top of upstream/main:
    ```bash
    git fetch upstream
    git rebase upstream/main
    ```
    - Resolve conflicts before pushing updates to your branch

7. **Address review feedback promptly**:
    - Use additional commits or amend history if requested
    - Keep the PR focused on a single logical change

### Maintainer Notes

- Maintainers with write access may create branches directly on this repo if:
    - The work needs CI secrets or protected environments
    - The change requires close collaboration among multiple maintainers

- Always clean up merged branches to keep the branch list tidy
- Branch protections and required reviews must be respected