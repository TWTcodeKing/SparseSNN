---
name: toLinux
description: >
  Upload modified project files to the remote Linux development server via scp.
  Use this skill whenever the user says "toLinux", "upload to linux", "sync to server",
  "push to remote", "send files to linux", or any variation of wanting to transfer
  changed files to the Linux machine. Also trigger when the user says "/toLinux".
  If the user just finished making code changes and mentions needing to test on the
  server or run something remotely, proactively suggest using this skill.
---

# toLinux — Upload Modified Files to Remote Linux Server

Transfer changed files from the local Windows workspace to the identical project
structure on the remote Linux server, so the user can run/test them there.

## Server Details

- **Host**: `twt@10.130.144.31`
- **Remote base path**: `/home/twt/SparseSNN/`
- **Local base path**: The SparseSNN project root (current working directory)

## Workflow

### 1. Determine which files to upload

**If the user provided specific file paths as arguments**, use those directly — skip auto-detection.

**Otherwise**, auto-detect changed files by running these three git commands and combining the results (deduplicated):

```bash
git diff --name-only
git diff --name-only --cached
git ls-files --others --exclude-standard
```

This covers unstaged changes, staged changes, and new untracked files.

### 2. Show the file list

Before uploading, list all files that will be transferred so the user can see what's going. Format like:

```
Uploading N files to twt@10.130.144.31:/home/twt/SparseSNN/
  - iengine/common/profiler.py
  - iengine/common/base.py
  ...
```

### 3. Upload via scp

For each file, run scp with the correct paths. Since the local machine is Windows,
convert any backslashes in paths to forward slashes before passing to scp.

**Important**: First ensure the remote directory exists, then copy the file.
Batch this efficiently — collect all unique parent directories and create them
in one ssh call, then scp the files.

```bash
# Create all needed remote directories in one shot
ssh twt@10.130.144.31 "mkdir -p /home/twt/SparseSNN/path/to/dir1 /home/twt/SparseSNN/path/to/dir2"

# Upload each file
scp "path/to/file" twt@10.130.144.31:/home/twt/SparseSNN/path/to/file
```

If there are many files (say 4+), you can batch them more efficiently using a
single scp call with multiple source files going to the same destination directory,
or by using `tar` piped through ssh:

```bash
tar cf - file1 file2 file3 | ssh twt@10.130.144.31 "cd /home/twt/SparseSNN && tar xf -"
```

### 4. Confirm completion

After all uploads finish, summarize what was transferred:

```
Done! Uploaded N files to twt@10.130.144.31:/home/twt/SparseSNN/
```

## Path handling notes

- Git outputs paths with forward slashes even on Windows — use those directly for
  the remote path portion.
- If the user passes Windows-style paths (with backslashes), convert them to
  forward slashes before use.
- All paths are relative to the project root. Never use absolute local paths in
  the scp destination.
