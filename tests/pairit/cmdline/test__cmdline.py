import shutil
import subprocess

import pytest


# Only pairit_train is exercised in the fast CI: each subprocess reimports the
# full pairit + torch + rdkit + hydra stack (~500 MB peak), and running all
# three back-to-back consistently OOMs the current kubernetes runner. Since
# all three CLIs share identical import machinery, one is sufficient to
# verify the entry-point wiring works. See MR!11593 for the OOM history.
@pytest.mark.parametrize("script", ["pairit_train"])
def test_hydra_entrypoint_help(script):
    if shutil.which(script) is None:
        pytest.skip(f"{script} is not installed in this environment")

    output = subprocess.check_output([script, "-h"]).splitlines()

    # Some import-time prints (e.g. PRETRAINED_TOKENIZER_PATH) pollute stdout
    # before Hydra's help header. Check ANY line, not strictly output[0].
    assert any(b"powered by Hydra." in line for line in output), (
        f"expected 'powered by Hydra.' in {script} -h output; got: {output[:5]}"
    )
    assert len(output) > 1
