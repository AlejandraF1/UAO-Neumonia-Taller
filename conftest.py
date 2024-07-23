# conftest.py

import pytest

# Registra la marca "slow"
pytest.mark.slow = pytest.mark.mark_custom(name="slow")

# Registra la marca "gpu"
pytest.mark.gpu = pytest.mark.mark_custom(name="gpu")

# Registra la marca "flaky"
pytest.mark.flaky = pytest.mark.mark_custom(name="flaky")

# Registra la marca "xfail_with_pyarrow_strings"
pytest.mark.xfail_with_pyarrow_strings = pytest.mark.mark_custom(name="xfail_with_pyarrow_strings")
