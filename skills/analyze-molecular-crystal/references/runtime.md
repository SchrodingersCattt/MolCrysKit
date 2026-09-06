# Runtime Recovery

Read only after `mck` is missing or a documented command reports a
version/capability error.

```bash
python -m pip install "molcrys-kit==0.7.1"
mck --version
```

Use the same interpreter for installation and execution. Do not reinstall,
upgrade dependencies, inspect package internals, or probe unrelated commands.
After installation, retry the original command once. If it still fails, preserve
the exact error and stop changing the environment.
