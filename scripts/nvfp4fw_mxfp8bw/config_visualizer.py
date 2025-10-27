from pprint import pprint
from fiddle import printing
import fiddle.graphviz as fgv
import os

class ConfigVisualizer:

    def __init__(self, cfg, outdir=None):
        self.cfg = cfg
        self.grey = '\033[90m'
        self.reset = '\033[0m'
        self.outdir = outdir
        if outdir is not None:
            os.makedirs(outdir, exist_ok=True)

    def native_print(self):
        print(printing.as_str_flattened(self.cfg))


    def print_all(self, label="config_viz"):
        lines = printing.as_str_flattened(self.cfg).split("\n")
        if self.outdir is not None:
            f = open(f"{self.outdir}/{label}_all.txt", "w")
                
            for line in lines:
                if "unset" in line:
                    print(f"{self.grey}{line}{self.reset}")
                    if self.outdir is not None:
                        f.write(f"{self.grey}{line}{self.reset}\n")
                else:
                    print(line)
                    if self.outdir is not None:
                        f.write(f"{line}\n")
        if self.outdir is not None:
            f.close()

    def print(self):
        pprint(printing.as_dict_flattened(self.cfg))

    def draw(self, label="config_viz"):
        if self.outdir is not None:
            output_path = f"{self.outdir}/svg.{label}"
            g = fgv.render(self.cfg)
            g.render(output_path, format="svg", cleanup=True)
            print(f"Config graph saved to {output_path}.svg")
        else:
            print("[Warning] Output directory not set, skip drawing config graph.")