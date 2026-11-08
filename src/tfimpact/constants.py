from __future__ import annotations

import re

# Terraform block kinds
RESOURCE = "resource"
DATA = "data"
MODULE = "module"
VARIABLE = "variable"
OUTPUT = "output"
LOCAL = "local"
BLOCK_KINDS = (RESOURCE, DATA, MODULE, VARIABLE, OUTPUT, LOCAL)

# address prefix per kind (address = f"{dir}::{prefix}.{...}")
KIND_PREFIX = {
    RESOURCE: "resource",
    DATA: "data",
    MODULE: "module",
    VARIABLE: "var",
    OUTPUT: "output",
    LOCAL: "local",
}

ROOT_DIR = "."
ADDR_SEP = "::"

# change actions in git diff
ADDED = "+"
REMOVED = "-"
MODIFIED = "~"
ACTIONS = (ADDED, REMOVED, MODIFIED)

TF_SUFFIX = ".tf"

# git object modes we accept as regular files
GIT_REGULAR_MODES = ("100644", "100755")
NULL_OID = "0" * 40

# resource type shape (provider-agnostic)
RESOURCE_TYPE_RE = re.compile(r"^[a-z0-9]+_[a-z0-9_]+$")

# commit skip reasons
SKIP_START = "start_commit"
SKIP_MERGE = "merge"
SKIP_NO_TF = "no_tf_change"
SKIP_TOO_FEW = "too_few_items"
SKIP_TOO_MANY = "too_many_items"
