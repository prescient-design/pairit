import shutil
import subprocess

import pytest


# Only pairit_train is exercised here: each subprocess reimports the full
# pairit + torch + rdkit + hydra stack (~500 MB peak), and running all three
# back-to-back exhausts memory on smaller CI runners. All three CLIs share the
# same import machinery, so one is enough to verify the entry-point wiring.
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
