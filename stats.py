import json

def main():
    with open("stats_sepsis_cut_0.json") as f:
        stats = json.load(f)
        print(f"Avg trad q: {sum(stats["trad_q"])/len(stats["trad_q"])}")
        print(f"Avg trad v: {sum(stats["trad_v"])/len(stats["trad_v"])}")
        print(f"Avg unf q: {sum(stats["unf_q"])/len(stats["unf_q"])}")
        print(f"Avg unf v: {sum(stats["unf_v"])/len(stats["unf_v"])}")

if __name__ == "__main__":
    main()