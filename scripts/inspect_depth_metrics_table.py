import json

with open("reports/argo2020/argo2020_metrics.json") as f:
    d = json.load(f)

print(f"{'Depth':>8s} | {'Count':>6s} | {'Climatology':>12s} | {'Pointwise MLP':>14s} | {'Simple CNN':>12s} | {'OceanEmbed':>12s}")
print("-" * 74)
for depth_str, stats in d["depth_wise_metrics"].items():
    cnt = stats["count"]
    if cnt == 0:
        continue
    c_rmse = f"{stats['climatology']['rmse']:.4f}" if stats['climatology']['rmse'] else 'N/A'
    m_rmse = f"{stats['mlp']['rmse']:.4f}" if stats['mlp']['rmse'] else 'N/A'
    cnn_rmse = f"{stats['simple_cnn']['rmse']:.4f}" if stats['simple_cnn']['rmse'] else 'N/A'
    oe_rmse = f"{stats['oceanembed']['rmse']:.4f}" if stats['oceanembed']['rmse'] else 'N/A'
    print(f"{depth_str:>8s} | {cnt:>6d} | {c_rmse:>12s} | {m_rmse:>14s} | {cnn_rmse:>12s} | {oe_rmse:>12s}")
