# AudioX WebUI 维护回溯（todo）

> 仓库：`https://github.com/kegeai888/AudioXwebui`
> 最近关键提交：`5d8568d`（WebUI 修复/优化与文档同步）

## 1. 本轮目标（已完成）

- [x] 修复 `start_app.sh` 启动链路中的环境依赖报错
- [x] 排查并修复 WebUI 推理阶段关键错误
- [x] 统一上传视频预览与输出视频预览样式（居中、最大宽度 600px）
- [x] 输出文件改为保存到 `outputs/`，采用时间戳防重名命名
- [x] 修复“生成成功但视频预览不显示”问题
- [x] 更新文档与仓库地址并完成提交推送

---

## 2. 二次开发与优化清单（按文件）

### `audiox/interface/gradio.py`
- [x] 音频保存链路增强：`torchaudio.save` + `soundfile` 回退，规避 `torchcodec` 依赖导致的失败
- [x] 输出目录由 `demo_result/` 改为 `outputs/`
- [x] 输出命名规则改为：`outputs_YYYYMMDDHHMMSS(.wav/.mp4)`，同秒冲突自动加 `_1/_2...`
- [x] 上传视频预览改为 HTML `<video>` 渲染，避免 Gradio 组件样式受限
- [x] 输出视频预览与上传预览统一样式（居中 + `max-width:600px`）
- [x] `generate_cond` 返回值调整为 `(output_video_html, audio_path)` 以匹配前端展示
- [x] 新增路径/预览辅助函数，减少 UI 逻辑分散

### `run_gradio.py`
- [x] `create_ui(..., space_like=args.space_like_ui)` 参数贯通
- [x] `launch()` 显式设置：
  - `server_name="0.0.0.0"`
  - `server_port=7860`
  - `allowed_paths=[os.path.abspath("outputs")]`（修复浏览器无法读取本地输出文件）
- [x] 新增 `--space-like-ui` 参数

### `setup.py`
- [x] 增加依赖：`timm`

### `README.md`
- [x] 仓库地址更新为：`https://github.com/kegeai888/AudioXwebui.git`
- [x] Clone 指令同步更新

### `用户使用手册.md`
- [x] 顶部仓库地址更新
- [x] 补充上传/输出视频预览规则（居中、最大宽度 600px）
- [x] 新增输出文件保存规则说明（`outputs/` + 时间戳 + 防重名）
- [x] 补充 FAQ：生成成功但视频预览未显示的排查项

### `.gitignore`
- [x] 增加忽略：`cc.sh`、`outputs/`
- [x] 继续忽略模型/大文件（如 `model/`、`*.ckpt`、`*.wav` 等）

---

## 3. 关键问题复盘（现象 -> 根因 -> 修复）

### 问题 A：启动时报 `numpy/scipy` 兼容错误
- 现象：`All ufuncs must have type numpy.ufunc`
- 根因：环境中 `numpy` 与 `scipy` 二进制/版本不匹配
- 修复：重装并固定可用组合（如 `numpy==1.23.5` + `scipy==1.14.1`）
- 经验：科学计算栈优先“成组锁版本”

### 问题 B：推理结束保存音频失败（`torchcodec`）
- 现象：`ImportError: TorchCodec is required for save_with_torchcodec`
- 根因：当前 `torchaudio.save` 后端路径触发了 `torchcodec` 强依赖
- 修复：优先 `torchaudio.save(..., backend="soundfile")`，失败回退 `soundfile.write`
- 经验：I/O 链路必须有可用降级路径

### 问题 C：视频预览尺寸规则不生效
- 现象：上传预览仍过大，CSS 不稳定
- 根因：`gr.Video` 内部 DOM/样式优先级导致外部 CSS 难以稳定覆盖
- 修复：改为 `gr.HTML` + 原生 `<video>` + 内联样式
- 经验：复杂组件样式不可控时，改原生渲染更稳

### 问题 D：生成视频文件存在但页面不显示
- 现象：`outputs/*.mp4` 已生成，前端仍无预览
- 根因：Gradio 默认不允许访问该目录文件
- 修复：`launch(..., allowed_paths=[abs(outputs)])`
- 经验：文件可见性问题优先检查服务端路径白名单

---

## 4. Git/GitHub 发布记录

- [x] 配置本仓库提交身份：
  - `user.name = kegeai888`
  - `user.email = ihuangke222@gmail.com`
- [x] `gh auth login --web` 完成设备码浏览器认证
- [x] 目标远端：`https://github.com/kegeai888/AudioXwebui.git`
- [x] 推送分支：`main`
- [x] 已推送提交：`5d8568d`

---

## 5. 维护经验与准则（后续改动建议）

- [x] 优先保证“可运行 + 可回退”，避免单点依赖
- [x] UI 预览问题先分离为：数据路径、服务权限、前端渲染 三层排查
- [x] 输出文件统一目录与命名规范，减少历史残留覆盖
- [x] 文档必须与行为同步更新（README + 使用手册）
- [x] 推送前只提交必要文件，持续过滤大文件/临时文件

---

## 6. 回归检查清单（每次发布前后执行）

### 启动前
- [ ] `python -c "import scipy, librosa"` 无异常
- [ ] 模型路径与依赖完整（`model/`、`timm`）

### 运行时
- [ ] 文本生成音频可成功输出 `outputs/*.wav`
- [ ] 带视频条件可成功输出 `outputs/*.mp4`
- [ ] 上传视频预览居中且宽度不超过 600px
- [ ] 输出视频预览与上传样式一致

### 发布前
- [ ] `git status` 仅包含预期文件
- [ ] `cc.sh`、`outputs/`、模型大文件未被纳入提交
- [ ] 文档链接与命令已同步

---

## 7. 待办（后续可选）

- [ ] 决定是否将 `start_app.sh` 纳入版本管理
- [ ] 决定是否保留项目内 `CLAUDE.md`（团队规范与自动化规则）
- [ ] 增加最小化自动化回归脚本（启动/推理/输出检查）
