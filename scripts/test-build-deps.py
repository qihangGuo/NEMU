#!/usr/bin/env python3
"""Host regressions for NEMU's C/C++ dependency tracking; no guest execution."""

import ctypes
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


NEMU_HOME = Path(__file__).resolve().parents[1]
BUILD_ENV = dict(os.environ, NEMU_HOME=str(NEMU_HOME), CCACHE_DISABLE="1")


class BuildDependenciesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        subprocess.run(["make", "-s", "-C", str(NEMU_HOME / "tools/fixdep")],
                       env=BUILD_ENV, check=True)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="nemu-build-deps-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "original"
        self.root.mkdir()
        (self.root / "include").mkdir()
        (self.root / ".config").touch()
        (self.root / "Makefile").write_text(
            f"NEMU_HOME := {NEMU_HOME}\n"
            "NAME := deps-test\nSHARE := 1\nCC := gcc\nCXX := g++\n"
            "SRCS := producer.c\nXSRCS := consumer.cpp\n"
            "INC_DIR := $(CURDIR)/include\n"
            "include $(NEMU_HOME)/scripts/config.mk\n"
            "include $(NEMU_HOME)/scripts/build.mk\n"
        )
        self.write_header(8)
        (self.root / "producer.c").write_text(
            '#include "state.h"\n'
            "struct CpuState cpu;\n"
            "size_t producer_offset(void) { return offsetof(struct CpuState, priorities); }\n"
            "void init_cpu(void) { static int priority = 42; cpu.priorities = &priority; }\n"
        )
        (self.root / "consumer.cpp").write_text(
            '#include "state.h"\n'
            'extern "C" {\n'
            "extern struct CpuState cpu;\n"
            "size_t consumer_offset(void) { return offsetof(struct CpuState, priorities); }\n"
            "int read_priority(void) { return *cpu.priorities; }\n"
            "}\n"
        )
        self.build()

    def write_header(self, words):
        (self.root / "include/state.h").write_text(
            "#include <stdint.h>\n#include <stddef.h>\n"
            f"struct CpuState {{ uint64_t evu[{words}]; int *priorities; }};\n"
        )

    def build(self):
        result = subprocess.run(
            ["make", "-s", "-j2", "app"], cwd=self.root, env=BUILD_ENV,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout)

    def object_times(self):
        return {p.name: p.stat().st_mtime_ns for p in (self.root / "build").rglob("*")
                if p.suffix in (".o", ".opp")}

    def check_layout(self, words):
        library = ctypes.CDLL(str(self.root / "build/deps-test-so"))
        library.producer_offset.restype = ctypes.c_size_t
        library.consumer_offset.restype = ctypes.c_size_t
        self.assertEqual(library.producer_offset(), words * 8)
        # Check layout before dereferencing: stale objects would read a bad pointer.
        self.assertEqual(library.consumer_offset(), library.producer_offset())
        library.init_cpu()
        self.assertEqual(library.read_priority(), 42)

    def test_relocated_build(self):
        relocated = Path(self.temp.name) / "relocated"
        shutil.copytree(self.root, relocated)
        self.root = relocated
        # Match rsync: old .d files contain the original absolute object paths.
        self.write_header(23)
        producer = self.root / "producer.c"
        producer.write_text(producer.read_text() + "\n/* Updated initialization. */\n")
        self.build()
        self.check_layout(23)
        before = self.object_times()
        self.build()
        self.assertEqual(self.object_times(), before, "relocated objects rebuild repeatedly")

    def test_missing_dependencies(self):
        for dependency in (self.root / "build").rglob("*.d"):
            dependency.unlink()
        self.write_header(23)
        self.build()
        self.check_layout(23)

    def test_header_change_and_incremental_build(self):
        before = self.object_times()
        self.build()
        self.assertEqual(self.object_times(), before, "unchanged objects were rebuilt")
        self.write_header(23)
        self.build()
        self.check_layout(23)
        self.assertTrue(all(self.object_times()[name] > stamp for name, stamp in before.items()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
