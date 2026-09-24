# Introduction

Research systems programming language focused on concurrency and memory safety.

**Linux x86_64 only for now.**

```rs
package input;

import zeta::io;
import zeta::file.File;
import zeta::file.FileWriter;
import zeta::file.FileReader;
import zeta::result.Result;
import zeta::io_error.IOError;

struct Point {
    x: i64,
    y: i64,
}

struct Line {
    start: Point,
    end: Point,
}

func main(): i64 {
    io.stdout().writeln("=== uninit Success Suite ===");

    let mut scalar: i64 = uninit;
    scalar = 42;
    if (scalar == 42) {
        io.stdout().writeln("[PASS] scalar uninit write-then-read");
    } else {
        io.stdout().writeln("[FAIL] scalar value wrong (unreachable)");
        return 1;
    }

    let mut branchy: i64 = uninit;
    if (scalar == 42) {
        branchy = 10;
    } else {
        branchy = 20;
    }
    
    if (branchy == 10) {
        io.stdout().writeln("[PASS] conditional uninit merge (both arms init)");
    } else {
        io.stdout().writeln("[FAIL] unexpected branch result");
        return 1;
    }

    let mut p: Point = uninit;
    p.x = 1;
    p.y = 2;
    let sum: i64 = p.x + p.y;
    if (sum == 3) {
        io.stdout().writeln("[PASS] struct field-level uninit tracking");
    } else {
        io.stdout().writeln("[FAIL] struct field sum wrong");
        return 2;
    }

    let mut line: Line = uninit;
    line.start.x = 0;
    line.start.y = 0;
    line.end.x = 5;
    line.end.y = 5;
    
    let dx: i64 = line.end.x - line.start.x;
    let dy: i64 = line.end.y - line.start.y;
    if (dx == 5 && dy == 5) {
        io.stdout().writeln("[PASS] nested struct uninit tracking");
    } else {
        io.stdout().writeln("[FAIL] nested struct math wrong");
        return 3;
    }

    
    let mut arr: [4]i64 = uninit;
    arr[0] = 100;
    arr[1] = 200;
    arr[2] = 300;
    arr[3] = 400;
    
    let a0: i64 = arr[0];
    let a3: i64 = arr[3];
    if (a0 == 100 && a3 == 400) {
        io.stdout().writeln("[PASS] array uninit per-element tracking");
    } else {
        io.stdout().writeln("[FAIL] array element mismatch");
        return 4;
    }

    let ar: &mut i64 = &mut arr[2];
    *ar = 30;
    io.stdout().writeln("[PASS] mutable reborrow into an already-initialized array slot");

    let mut cell: i64 = uninit;
    fill(&mut cell);
    if (cell == 7) {
        io.stdout().writeln("[PASS] &mut borrow into uninit storage, initialized by callee");
    } else {
        io.stdout().writeln("[FAIL] fill() did not set expected value");
        return 5;
    }

    let mut owner: Point = uninit;
    owner.x = 9;
    owner.y = 9;
    let moved: Point = owner;
    
    if (moved.x == 9 && moved.y == 9) {
        io.stdout().writeln("[PASS] move out of a fully-initialized uninit-declared value");
    } else {
        io.stdout().writeln("[FAIL] moved struct fields wrong");
        return 6;
    }

    let mut counter: i64 = uninit;
    counter = 1;
    counter = 2;
    if (counter == 2) {
        io.stdout().writeln("[PASS] reinitializing an already-initialized place");
    } else {
        io.stdout().writeln("[FAIL] reassignment failed");
        return 7;
    }

    let mut f: File = match (File.create("./build/uninit_test.txt")) {
        case Result.Ok { value } -> value,
        case Result.Err { error } -> {
            io.stdout().writeln("[FAIL] could not create file for uninit I/O test");
            return 8;
        },
    };
    let mut writer: FileWriter = f.writer();
    let write_res: Result<usize, IOError> = writer.write_all("uninit-io-check".as_bytes());
    let close_res: Result<void, IOError> = f.close();

    let mut rf: File = match (File.open("./build/uninit_test.txt")) {
        case Result.Ok { value } -> value,
        case Result.Err { error } -> {
            io.stdout().writeln("[FAIL] could not reopen file for uninit I/O test");
            return 9;
        },
    };
    let mut io_buf: [16]u8 = uninit;
    let mut reader: FileReader = rf.reader();
    let n: usize = match (unsafe { reader.read_raw(&mut io_buf[0..<16]) }) {
        case Result.Ok { value } -> value,
        case Result.Err { error } -> 0,
    };

    io.stdout().writeln("=== All uninit checks passed ===");
    return 0;
}

func fill(x: &mut i64) {
    *x = 7;
}

func consume_pair(a: &mut i64, b: &mut i64) {
}
```

### Why Zeta?
- Reliability: Ergonomic memory safety without a garbage collector like you have never seen
- Performance: Build high performance no-gc applications that compile to machine code and utilize zero cost abstractions

# Install Zeta (Linux)

The recommended way to install Zeta is through the official release archive.

The archive contains the complete Zeta toolchain:

* `zeta-lang` -- Zeta compiler
* `zeta-lsp` -- Language server for editor integrations
* `zetaup` -- Zeta toolchain updater
* `lib/` -- Zeta standard library

## Quick Install

Download the latest release:

```bash
curl -LO https://github.com/Voxon-Development/zeta-lang/releases/latest/download/zeta-linux-x86_64.tar.gz
```

Extract it:

```bash
tar -xzf zeta-linux-x86_64.tar.gz
cd zeta-linux-x86_64
```

Install Zeta:

```bash
mkdir -p ~/.local/bin
mkdir -p ~/.local/share/zeta

cp zeta-lang zeta-lsp zetaup ~/.local/bin/
cp -r lib ~/.local/share/zeta/
```

Add `~/.local/bin` to your `PATH` if needed:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

Verify your installation:

```bash
zeta-lang --version
zeta-lsp --version
zetaup --version
```

You are now ready to use Zeta.

---

## Updating Zeta

Zeta includes a toolchain updater that keeps the compiler, language server, and standard library synchronized.

Run:

```bash
zetaup upgrade
```

The updater will:

* Check for the latest Zeta release
* Download updated binaries
* Update `zeta-lang`
* Update `zeta-lsp`
* Update the standard library

To see more details:

```bash
zetaup --verbose upgrade
```

**zetaup must be updated manually using the steps above, zetaup cannot update itself.

---

# Building from Source (Linux)

Building from source is intended for contributors and compiler developers.

## Prerequisites

* A recent nightly Rust toolchain: (Zeta depends on stuff such as allocator_api)

```bash
rustup install nightly
```

* `git`

## Clone and Build

```bash
git clone https://github.com/Voxon-Development/zeta-lang.git
cd zeta-lang

cargo build --release --bin zeta-lang
cargo build --release --bin zeta-lsp
cargo build --release --bin zetaup
```

This produces:

```text
target/release/
--- zeta-lang
--- zeta-lsp
--- zetaup
```

You can install the locally built binaries:

```bash
mkdir -p ~/.local/bin

cp target/release/zeta-lang ~/.local/bin/
cp target/release/zeta-lsp ~/.local/bin/
cp target/release/zetaup ~/.local/bin/
```

The standard library is available in the repository under:

```text
lib/
```

---

# Contributing

We’re excited you want to contribute to Zeta-Lang!

## How to Contribute

### Report Bugs

Open an issue on GitHub if you encounter a bug or unexpected behavior.

Please include:

* A minimal reproducible example
* Expected behavior
* Actual behavior
* Zeta version

### Submit Pull Requests

1. Fork the repository and create a feature branch:

```bash
git checkout -b feature/YourFeature
```

2. Write clear and documented code.

3. Include tests for new features or bug fixes.

4. Ensure existing tests pass before submitting.

### Code Style

Follow consistent formatting and naming conventions:

* Use `snake_case` for variables and functions.
* Use `PascalCase` for types.
* Keep formatting consistent.

### Documentation

Update documentation when adding features or changing behavior.

### Community Etiquette

Be respectful and collaborative.

Contributions of all sizes are welcome, from fixing typos to implementing major compiler features.
