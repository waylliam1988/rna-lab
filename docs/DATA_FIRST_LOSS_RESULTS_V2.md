# S6-L：BCE 与伙伴竞争损失 v2 结果

执行日期：2026-10-28。该报告只覆盖冻结的 `joint_split_v2` train/validation 开发对照；没有新独立测试集，旧 test 未访问。

协议：`data/manifests/data_first_loss_protocol_v2.json`，SHA256 `46db4520d0bcfc15319f96def77258b4c62ec612837d34caa2a281c098a02659`。v2 是对 v1 验证维度技术失败的修复后新版本；不复用 v1 checkpoint 或指标。两条件均为 150019 参数 CNN、550 次 AdamW 更新、seed 42/43/44、相同组均匀采样日程，唯一实验因素是 `lambda=0.05` 的伙伴竞争项。

六次运行和最终检查点均成功，独立重载验证报告为 `data/manifests/data_first_loss_verification_v2.json`，SHA256 `ca1117899d08c0bad03871688924d6ab9e1c9a51645827b01df55daa67d311ef`。重载后逐 RNA 解码、group-macro F1 和多伙伴诊断都与运行结果一致。

| 条件 | seed 42 | seed 43 | seed 44 | 平均 group-macro F1 | 平均训练秒 | 平均峰值显存 MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BCE | 0.511988 | 0.546002 | 0.509172 | 0.522387 | 14.319 | 88.948 |
| BCE + competition | 0.527559 | 0.554906 | 0.491957 | 0.524807 | 16.408 | 98.430 |

竞争项减 BCE 的 seed-level group-macro F1 差值为 `+0.015571`、`+0.008903`、`−0.017215`，平均 `+0.002420`。7 个 validation groups 中有 3 个改善；没有 group 下降超过预声明的 0.10，但三 seed 全部改善、平均提升至少 0.02 和至少 4/7 groups 改善这三个条件均未同时满足。预声明 development gate **不通过**。

摘要 manifest：`data/manifests/data_first_loss_summary_v2.json`，SHA256 `53493b48511023ddbb100cef68cb9c871168ba7648edcec0f501063711b53363`。结论按冻结规则为：伙伴竞争损失保留为已实现候选，但停止可选合入；不改变训练 CLI，不启用为默认；默认继续使用 balanced BCE。该结论只适用于这 88 条训练 RNA、10 条 validation RNA 和 7 个 validation groups，不能作为跨家族泛化证据。
