# mujofil-warp (deprecated alias)

This project was **renamed to [`mujofil`](https://pypi.org/project/mujofil/)** in
version 0.2.0.

Installing `mujofil-warp` now simply installs `mujofil` (it is a dependency-only
alias package). Please switch to:

```bash
pip install mujofil
```

```python
import mujofil
```

For backward compatibility, `mujofil` still provides a thin `mujofil_warp` module
that re-exports everything with a deprecation warning, so existing
`import mujofil_warp` code keeps working.

Source and documentation: https://github.com/tau-intelligence/mujofil
