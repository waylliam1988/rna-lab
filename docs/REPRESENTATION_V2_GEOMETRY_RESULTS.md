# 输入v2：geometry结果

目标、划分、种子、预算和模型尺寸固定；表中主误差为四个未来间隔等权的组宏平均。仅5个开发验证组，无独立测试。

| 模型 | 输入条件 | 主误差均值 ±种子标准差 |
| --- | --- | --- |
| ghost | drop_static_no_geom | 3.578038 ± 0.0916 |
| ghost | keep_static_no_geom | 3.461327 ± 0.125 |
| mlp | drop_static_no_geom | 3.357662 ± 0.0785 |
| mlp | keep_static_no_geom | 3.476388 ± 0.184 |
| ghost | drop_static_geom | 3.508738 ± 0.176 |
| ghost | keep_static_geom | 3.341621 ± 0.162 |
| mlp | drop_static_geom | 2.915638 ± 0.195 |
| mlp | keep_static_geom | 3.025168 ± 0.156 |

## 配对贡献

正收益表示新条件误差降低，负值表示误差增加；drop_static的“新条件”是移除第2/3项，add_geometry的新条件是添加实时几何接触。

| 模型 | 比较 | 平均配对误差下降 | 新条件较好种子 | 较好组 | 达到预声明较一致收益规则 |
| --- | --- | --- | --- | --- | --- |
| ghost | drop_static_without_geometry | -3.43% | 0/3 | 0/5 | False |
| mlp | drop_static_without_geometry | 3.20% | 2/3 | 2/5 | False |
| ghost | add_geometry_static_kept | 3.49% | 3/3 | 3/5 | False |
| mlp | add_geometry_static_kept | 12.86% | 3/3 | 2/5 | False |
| ghost | add_geometry_static_dropped | 1.99% | 2/3 | 2/5 | False |
| mlp | add_geometry_static_dropped | 13.20% | 3/3 | 4/5 | True |
| ghost | drop_static_with_geometry | -5.05% | 0/3 | 2/5 | False |
| mlp | drop_static_with_geometry | 3.47% | 2/3 | 4/5 | False |

## 全部未来间隔

| 模型 | 输入条件 | 未来步数 | 标准化动态MSE | 对线性Skill |
| --- | --- | --- | --- | --- |
| ghost | drop_static_no_geom | 1 | 1.669862 | -2.6612 |
| ghost | drop_static_no_geom | 4 | 2.107344 | 0.2383 |
| ghost | drop_static_no_geom | 8 | 3.146884 | 0.1897 |
| ghost | drop_static_no_geom | 16 | 7.388062 | 0.2377 |
| ghost | keep_static_no_geom | 1 | 1.732367 | -2.7982 |
| ghost | keep_static_no_geom | 4 | 1.975395 | 0.2860 |
| ghost | keep_static_no_geom | 8 | 2.986767 | 0.2310 |
| ghost | keep_static_no_geom | 16 | 7.150781 | 0.2621 |
| mlp | drop_static_no_geom | 1 | 1.382017 | -2.0301 |
| mlp | drop_static_no_geom | 4 | 1.996585 | 0.2783 |
| mlp | drop_static_no_geom | 8 | 2.987887 | 0.2307 |
| mlp | drop_static_no_geom | 16 | 7.064159 | 0.2711 |
| mlp | keep_static_no_geom | 1 | 1.608314 | -2.5262 |
| mlp | keep_static_no_geom | 4 | 2.046863 | 0.2602 |
| mlp | keep_static_no_geom | 8 | 3.051363 | 0.2143 |
| mlp | keep_static_no_geom | 16 | 7.199012 | 0.2572 |
| ghost | drop_static_geom | 1 | 1.678675 | -2.6805 |
| ghost | drop_static_geom | 4 | 2.142551 | 0.2256 |
| ghost | drop_static_geom | 8 | 2.991274 | 0.2298 |
| ghost | drop_static_geom | 16 | 7.222455 | 0.2547 |
| ghost | keep_static_geom | 1 | 1.581734 | -2.4680 |
| ghost | keep_static_geom | 4 | 1.975063 | 0.2861 |
| ghost | keep_static_geom | 8 | 2.89907 | 0.2535 |
| ghost | keep_static_geom | 16 | 6.910617 | 0.2869 |
| mlp | drop_static_geom | 1 | 1.058365 | -1.3205 |
| mlp | drop_static_geom | 4 | 1.643491 | 0.4060 |
| mlp | drop_static_geom | 8 | 2.53938 | 0.3462 |
| mlp | drop_static_geom | 16 | 6.421315 | 0.3374 |
| mlp | keep_static_geom | 1 | 1.293462 | -1.8359 |
| mlp | keep_static_geom | 4 | 1.650253 | 0.4035 |
| mlp | keep_static_geom | 8 | 2.584392 | 0.3346 |
| mlp | keep_static_geom | 16 | 6.572564 | 0.3218 |

## 几何接触增量的逐组差异

先平均种子的每组主误差，再计算相对下降；正数为改善，负数为退步。与全体平均配对下降的聚合顺序不同，不混算。

| 模型 | 条件 | 验证构建体 | 组误差下降 |
| --- | --- | --- | --- |
| ghost | add_geometry_static_kept | 7LYJ_A | -1.92% |
| ghost | add_geometry_static_kept | 3D0U_A | 4.58% |
| ghost | add_geometry_static_kept | 5KPY_A | 7.62% |
| ghost | add_geometry_static_kept | 4XW7_A | 11.95% |
| ghost | add_geometry_static_kept | 7K16_A | -17.22% |
| mlp | add_geometry_static_kept | 7LYJ_A | -11.64% |
| mlp | add_geometry_static_kept | 3D0U_A | 16.24% |
| mlp | add_geometry_static_kept | 5KPY_A | -4.37% |
| mlp | add_geometry_static_kept | 4XW7_A | 8.84% |
| mlp | add_geometry_static_kept | 7K16_A | -20.82% |
| ghost | add_geometry_static_dropped | 7LYJ_A | -11.87% |
| ghost | add_geometry_static_dropped | 3D0U_A | -0.44% |
| ghost | add_geometry_static_dropped | 5KPY_A | 29.95% |
| ghost | add_geometry_static_dropped | 4XW7_A | 37.40% |
| ghost | add_geometry_static_dropped | 7K16_A | -13.89% |
| mlp | add_geometry_static_dropped | 7LYJ_A | 17.75% |
| mlp | add_geometry_static_dropped | 3D0U_A | 14.16% |
| mlp | add_geometry_static_dropped | 5KPY_A | -0.59% |
| mlp | add_geometry_static_dropped | 4XW7_A | 8.89% |
| mlp | add_geometry_static_dropped | 7K16_A | 3.97% |

初始权重逐种子配对一致，训练查询顺序和参数量一致；移除通道采用标准化后整列置0，不移除索引12/13静态摘要。没有恢复J门控。几何接触只进入输入，预测目标/损失不增加。原量纲逐通道、逐组/种子、线性基线及权重哈希见各JSON。未检测到收益不等于普遍无用，也不等于统计等价。
