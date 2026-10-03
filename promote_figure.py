import json
import os
import shutil
from datetime import date


def promote_figure(name, source_png, params, caption="", seed=None):
    out_dir = os.path.join("thesis_figures", name)
    os.makedirs(out_dir, exist_ok=True)

    shutil.copy(source_png, os.path.join(out_dir, "plot.png"))

    config_record = {
        "params": params,
        "seed": seed,
        "date_generated": str(date.today()),
        "source_figure": source_png,
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config_record, f, indent=2, default=str)

    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write(f"# {name}\n\n")
        f.write(f"![plot](plot.png)\n\n")
        f.write(f"{caption}\n\n")
        f.write("**Exact configuration used:** see `config.json` in this folder.\n")

    print(f"promoted '{source_png}' -> thesis_figures/{name}/")