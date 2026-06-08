# 工作日志：离散 Fréchet 距离对比

**日期**: 2026-06-08

## 背景

项目原有标准 Hausdorff 和 Modified Hausdorff 都把旋律曲线视为点集，适合衡量几何覆盖关系，但不保留旋律点的先后顺序。为了与顺序敏感的曲线距离做对比，本次加入离散 Fréchet 距离。

## 本次改动

1. 在 `src/processing/hausdorff.py` 增加 `frechet_discrete(a_points, b_points)`。
   - 输入为两个有序点数组，形状分别为 `(N, D)` 和 `(M, D)`。
   - 使用动态规划计算离散 Fréchet 距离。
   - 空点集或非二维数组会抛出 `ValueError`。

2. 在 `src/analysis/distance_matrix.py` 接入 `method="frechet"`。
   - 可与 `standard`、`modified` 一样生成 N×N 对称距离矩阵。

3. 在 `src/ui/control_bar.py` 增加算法选项“离散 Fréchet”。
   - 修正距离算法选择逻辑，改为显式 label -> method 映射。

4. 更新 README 的核心功能描述。

5. 补充测试。
   - 相同曲线距离为 0。
   - 单点距离等于欧氏距离。
   - 反向序列在 Hausdorff 下距离为 0，但在 Fréchet 下距离大于 0，验证顺序敏感性。
   - 距离矩阵支持 `method="frechet"`。

## 算法说明

离散 Fréchet 距离可理解为两条曲线按顺序同步行走时，需要允许的最大牵引距离。动态规划递推为：

```text
ca[i,j] = max(
    dist(P[i], Q[j]),
    min(ca[i-1,j], ca[i-1,j-1], ca[i,j-1])
)
```

最终距离为 `ca[-1,-1]`。

## 与 Hausdorff 的区别

- Hausdorff：只看点集覆盖关系，不关心旋律顺序。
- Modified Hausdorff：用平均最近邻距离降低离群点影响，也不关心顺序。
- 离散 Fréchet：保留点序顺序，适合对照旋律走向是否一致。

## 验证

已运行：

```bash
pytest tests/test_hausdorff.py tests/test_distance_matrix.py -v --tb=short
```

结果：

```text
15 passed
```
