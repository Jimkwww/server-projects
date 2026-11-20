import json

a = json.load(open("results/interference_full12_merge_compile.json"))
b = json.load(open("results/interference_full12_merge_bf16.json"))

labels = ["AM_Jamming", "Comb_Jamming", "FM_Jamming", "Multi_Tone", "Narrowband",
          "No_Interference", "Protocol_Bluetooth", "Protocol_WiFi", "Pulse_DME",
          "Single_Tone", "Sweep_Chirp", "Wideband"]
print("%-22s %8s %8s %7s" % ("class", "merge", "compile", "diff"))
for l in labels:
    ca = a["per_class"][l]["acc"]
    cb = b["per_class"][l]["acc"]
    print("%-22s %8.4f %8.4f %+7.4f" % (l, cb, ca, ca - cb))
print("%-22s %8.4f %8.4f %+7.4f" % ("TOTAL", b["accuracy"], a["accuracy"], a["accuracy"] - b["accuracy"]))
