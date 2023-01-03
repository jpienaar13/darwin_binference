from binference.toymc_running import run_toymcs_from_cl
import sys

if __name__ == "__main__":
    print("Generating Toymc Run")
    run_toymcs_from_cl(sys.argv[1:])
