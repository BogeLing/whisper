"""
Whisper.cpp Python 封装
提供完整的 Python API 来使用 Homebrew 安装的 whisper-cli（Metal 加速）
"""

import os
import subprocess
import tempfile
import re
from pathlib import Path
from typing import Optional, List, Dict
import json
import threading
import time
from tqdm import tqdm
import ffmpeg


class WhisperCPP:
    """
    Whisper.cpp Python 封装类
    使用 Metal GPU 加速的 whisper-cli
    """
    
    def __init__(
        self, 
        model_name: str = "large-v3",
        model_dir: Optional[str] = None,
        threads: int = 10,
        language: str = "en"
    ):
        """
        初始化 WhisperCPP
        
        Args:
            model_name: 模型名称 (tiny, base, small, medium, large-v3)
            model_dir: 模型存储目录，默认为 ~/PycharmProjects/voiceRecognize/models
            threads: 使用的线程数，M4 芯片建议 10-12
            language: 转录语言，默认英语
        """
        self.model_name = model_name
        self.threads = threads
        self.language = language
        
        # 设置模型路径
        if model_dir is None:
            model_dir = os.path.expanduser("~/PycharmProjects/voiceRecognize/models")
        self.model_dir = Path(model_dir)
        self.model_path = self.model_dir / f"ggml-{model_name}.bin"
        
        # 确保模型存在
        self._ensure_model()
    
    def _ensure_model(self):
        """确保模型已下载"""
        if not self.model_path.exists():
            print(f"下载 {self.model_name} 模型...")
            self.model_dir.mkdir(parents=True, exist_ok=True)
            
            url = f"https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{self.model_name}.bin"
            subprocess.run(
                ["curl", "-L", url, "-o", str(self.model_path)],
                check=True,
                capture_output=True
            )
            print(f"✅ 模型下载完成: {self.model_path}")
    
    def _get_audio_duration(self, audio_path: str) -> float:
        """获取音频时长（秒）"""
        try:
            probe = ffmpeg.probe(audio_path)
            duration = float(probe["format"]["duration"])
            return duration
        except Exception as e:
            print(f"⚠️  无法获取音频时长: {e}")
            return 0
    
    def _convert_to_wav(self, audio_path: str) -> str:
        """
        将音频转换为 WAV 格式（whisper-cli 需要）
        
        Args:
            audio_path: 输入音频路径
            
        Returns:
            转换后的 WAV 文件路径
        """
        # 创建临时 WAV 文件
        temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_wav.close()
        
        # 使用 ffmpeg 转换
        subprocess.run([
            "ffmpeg", "-i", audio_path,
            "-ar", "16000",      # 16kHz 采样率
            "-ac", "1",          # 单声道
            "-c:a", "pcm_s16le", # 16-bit PCM
            "-y",                # 覆盖
            temp_wav.name
        ], check=True, capture_output=True)
        
        return temp_wav.name
    
    def transcribe(
        self,
        audio_path: str,
        output_format: str = "txt",
        verbose: bool = True
    ) -> Dict[str, any]:
        """
        转录音频文件（使用 Metal GPU 加速）
        
        Args:
            audio_path: 输入音频文件路径
            output_format: 输出格式 (txt, srt, vtt, json)
            verbose: 是否显示详细输出
            
        Returns:
            包含转录结果的字典
        """
        audio_path_obj = Path(audio_path).resolve()
        audio_path = str(audio_path_obj)
        
        # 1. 设定持久化 JSON 路径 (新建子文件夹)
        # 例如输入 video.mp4 -> 生成 video_output/video.json
        output_dir = audio_path_obj.parent / f"{audio_path_obj.stem}_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_base = str(output_dir / audio_path_obj.stem)
        json_file = f"{output_base}.json"
        
        # 2. 检查缓存：如果 JSON 已存在，直接使用
        if os.path.exists(json_file):
            if verbose:
                print(f"✨ 发现已有 JSON 缓存: {json_file}")
                print(f"⏩ 跳过 AI 推理，直接进行智能分段...")
            
            if output_format in ["srt", "txt"]:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                processed_text = self._smart_process(data, output_format)
                
                # 更新输出文件（覆盖旧的 txt/srt）
                output_file = f"{output_base}.{output_format}"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(processed_text)
                
                return {
                    'text': processed_text,
                    'output_file': output_file,
                    'success': True,
                    'cached': True
                }

        if verbose:
            print(f"🎤 转录音频: {audio_path}")
            print(f"📦 使用模型: {self.model_name} (Metal 加速)")
        
        # 3. 正常流程：转换为 WAV
        if verbose:
            print("🔄 转换音频格式...")
        temp_wav = self._convert_to_wav(audio_path)
        
        try:
            # 构建命令 - 开启全功能 JSON 和单词级时间戳
            # 【注意】这里不再用 tempfile，而是直接输出到源文件同级目录
            cmd = [
                "whisper-cli",
                "-m", str(self.model_path),
                "-f", temp_wav,
                "-ojf",               # 【核心】强制输出 Full JSON
                "-of", output_base,   # 输出到同级目录
                "-t", str(self.threads),
                "-l", self.language,
                "-sow", "true",
                "-pp",
            ]
            
            # 同时保留用户请求的格式输出（如果不是 JSON）
            if output_format != "json":
                cmd.append(f"-o{output_format}")
            
            # 启动进程
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            stdout_output = ""
            if verbose:
                print(f"⚡ 开始转录 M4 核心全力加速中...")
                with tqdm(total=100, desc="转录进度", unit="%", ncols=80) as pbar:
                    last_progress = 0
                    # 实时读取 stderr 来获取进度
                    while True:
                        line = process.stderr.readline()
                        if not line and process.poll() is not None:
                            break
                        if line:
                            # whisper.cpp 进度格式: "whisper_full_with_state: progress =  XX%"
                            if "progress =" in line:
                                try:
                                    progress_match = re.search(r"progress\s*=\s*(\d+)%", line)
                                    if progress_match:
                                        current_progress = int(progress_match.group(1))
                                        if current_progress > last_progress:
                                            pbar.update(current_progress - last_progress)
                                            last_progress = current_progress
                                except Exception:
                                    pass
                    pbar.n = 100
                    pbar.refresh()
            
            # 等待完成并获取所有输出
            stdout_output, stderr_output = process.communicate()
            
            if process.returncode != 0:
                print(f"❌ 转录失败: {stderr_output}")
                return {
                    'text': None,
                    'output_file': None,
                    'success': False,
                    'error': stderr_output
                }

            # 兼容后续代码使用的 result 对象
            class Result:
                def __init__(self, stdout):
                    self.stdout = stdout
            result = Result(stdout_output)
            
            # 读取结果
            json_file = f"{output_base}.json"
            if os.path.exists(json_file):
                # 如果我们要进行智能处理，从 JSON 读取并重新生成
                if output_format in ["srt", "txt"]:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    processed_text = self._smart_process(data, output_format)
                    
                    # 写回文件覆盖默认生成的
                    output_file = f"{output_base}.{output_format}"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(processed_text)
                    
                    text = processed_text
                else:
                    output_file = f"{output_base}.{output_format}"
                    with open(output_file, 'r', encoding='utf-8') as f:
                        text = f.read()
                
                if verbose:
                    print(f"✅ 转录完成 (已进行智能语义分段)！")
                
                return {
                    'text': text,
                    'output_file': output_file,
                    'success': True,
                    'stdout': result.stdout
                }
            else:
                return {
                    'text': None,
                    'output_file': None,
                    'success': False,
                    'error': '输出文件未生成'
                }
        
        finally:
            # 只清理生成的原始 WAV 文件
            # 结果文件由调用者（如 transcribe_to_file）负责清理或移动
            if os.path.exists(temp_wav):
                try:
                    os.unlink(temp_wav)
                except:
                    pass
    
    def _smart_process(self, data: Dict, output_format: str) -> str:
        """
        基于单词级别时间戳的终极分段逻辑
        1. 摊平所有单词，无视原始片段
        2. 按照语义标点和长度上限，重新组装句子
        """
        # 尝试从 Full JSON 中获取所有单词
        all_words = []
        segments = data.get("transcription", [])
        
        for seg in segments:
            tokens = seg.get("tokens", [])
            for tk in tokens:
                text = tk.get("text", "")
                
                # 【强化清理】移除所有 Whisper 特殊标记
                text = re.sub(r'\[_?BEG_?\]|\[_?TT_\d+\]|\[_?EOT_?\]|\[_?SOT_?\]', '', text)
                
                if not text.strip():
                    continue
                
                # 【重要修复】获取时间戳，不进行暴力回退到 Segment
                # 如果 token 自身没有 offsets，说明可能是标点，这不应该继承整个 Segment 的结束时间
                tk_offsets = tk.get("offsets", {})
                start = tk_offsets.get("from")
                end = tk_offsets.get("to")
                
                # 如果当前词没有时间戳（如标点），暂时标记为 None，稍后插值
                all_words.append({
                    "text": text,
                    "start": start,
                    "end": end,
                    "seg_start_fallback": seg["offsets"]["from"], # 仅用于兜底
                    "seg_end_fallback": seg["offsets"]["to"] 
                })

        if not all_words:
            return ""

        # --- 时间戳修复/插值 (Linear Interpolation) ---
        for i in range(len(all_words)):
            word = all_words[i]
            
            # 1. 修复 Start
            if word["start"] is None:
                if i > 0:
                    # 紧接上一个词结束
                    word["start"] = all_words[i-1]["end"]
                else:
                    # 如果是第一个词，被迫使用 Segment 开始
                    word["start"] = word["seg_start_fallback"]
            
            # 2. 修复 End
            if word["end"] is None:
                if word["start"] is not None:
                     # 假设它是标点，持续时间极短，或者就等于 start
                     word["end"] = word["start"]
                else:
                     # 依然无法确定，稍后处理
                     pass

        # 二次遍历确保没有 None (针对连续缺失的情况)
        for i in range(len(all_words)):
            if all_words[i]["end"] is None:
                 # 向后寻找最近的有效 start
                 valid_next_start = None
                 for j in range(i+1, len(all_words)):
                     if all_words[j]["start"] is not None:
                         valid_next_start = all_words[j]["start"]
                         break
                 
                 if valid_next_start:
                     all_words[i]["end"] = valid_next_start
                     all_words[i]["start"] = valid_next_start # 挤压成瞬间
                 else:
                     # 确实是全段最后了，只能用 segment end
                     all_words[i]["end"] = all_words[i]["seg_end_fallback"]
                     if all_words[i]["start"] is None:
                         all_words[i]["start"] = all_words[i]["end"]

        if not all_words:
            return ""

        merged_segments = []
        current_words_buffer = []  # 改用列表暂存单词对象，方便回溯
        current_len = 0
        
        MAX_CHARS = 90      # 用户设定长度

        for i, word in enumerate(all_words):
            text = word["text"]
            w_len = len(text)
            
            # --- 长度预判 ---
            if current_len + w_len > MAX_CHARS:
                # 【触发回溯切分逻辑】
                split_index = -1
                
                # 倒序寻找最近的标点符号 (逗号、句号等)
                # 我们希望切分点不要太靠前（保留至少 1/3 的长度），否则第一行太短
                min_keep_len = int(len(current_words_buffer) * 0.4)
                
                for j in range(len(current_words_buffer) - 1, min_keep_len, -1):
                    w_text = current_words_buffer[j]["text"].strip()
                    # 检查单词结尾是否是标点
                    if w_text and w_text[-1] in ['.', '!', '?', '。', '！', '？', ',', '，', ':', ';']:
                        split_index = j
                        break
                
                if split_index != -1:
                    # 方案 A：找到了完美的标点切分点
                    seg1_words = current_words_buffer[:split_index+1]
                    seg2_words = current_words_buffer[split_index+1:]
                    
                    merged_segments.append({
                        "text": "".join([w["text"] for w in seg1_words]).strip(),
                        "start": seg1_words[0]["start"],
                        "end": seg1_words[-1]["end"]
                    })
                    
                    # 剩下的词 + 当前新词 组成下一句的开头
                    current_words_buffer = seg2_words + [word]
                    current_len = sum(len(w["text"]) for w in current_words_buffer)
                    continue
                
                else:
                    # 方案 B：没找到标点，只能硬切
                    # 但要做一个保护：如果当前词是标点，必须把它贴到上一行，不能让它作为新行开头
                    is_bad_start = text.strip() and text.strip()[0] in ['.', '!', '?', ',', '，', ':']
                    
                    if not is_bad_start:
                        # 正常硬切：Buffer 里的归上一行，当前词归下一行
                        if current_words_buffer:
                            merged_segments.append({
                                "text": "".join([w["text"] for w in current_words_buffer]).strip(),
                                "start": current_words_buffer[0]["start"],
                                "end": current_words_buffer[-1]["end"]
                            })
                        current_words_buffer = [word]
                        current_len = w_len
                        continue
                    else:
                        # 这是一个标点，虽然超长了，但必须强行塞进上一行（稍后可能会在 Loop 底部触发 Sentence End 切分）
                        pass

            # --- 正常追加 ---
            current_words_buffer.append(word)
            current_len += w_len
            
            # --- 语义完结直接切分 (Post-split) ---
            # 如果碰到了强结束标点 (. ? !)，并且长度适中（>15字），直接切分，不留着过年
            curr_str = "".join([w["text"] for w in current_words_buffer]).strip()
            is_strong_end = curr_str and curr_str[-1] in ['.', '!', '?', '。', '！', '？']
            
            if is_strong_end and current_len > 15:
                merged_segments.append({
                    "text": curr_str,
                    "start": current_words_buffer[0]["start"],
                    "end": current_words_buffer[-1]["end"]
                })
                current_words_buffer = []
                current_len = 0
        
        # 补上最后一段
        if current_words_buffer:
            merged_segments.append({
                "text": "".join([w["text"] for w in current_words_buffer]).strip(),
                "start": current_words_buffer[0]["start"],
                "end": current_words_buffer[-1]["end"]
            })

        # 格式化输出
        if output_format == "srt":
            return self._format_as_srt_from_words(merged_segments)
        else:
            return "\n".join([s["text"] for s in merged_segments])

    def _format_as_srt_from_words(self, segments: List[Dict]) -> str:
        """从自定义片段生成 SRT"""
        srt = ""
        for i, seg in enumerate(segments):
            # 【修正】JSON offsets 单位是毫秒 (ms)，不需要 x10
            start_ms = seg["start"]
            end_ms = seg["end"]
            
            def to_srt_time(total_ms):
                # 确保 total_ms 是整数
                total_ms = int(total_ms)
                h = total_ms // 3600000
                m = (total_ms % 3600000) // 60000
                s = (total_ms % 60000) // 1000
                ms = total_ms % 1000
                return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

            srt += f"{i+1}\n"
            srt += f"{to_srt_time(start_ms)} --> {to_srt_time(end_ms)}\n"
            srt += f"{seg['text']}\n\n"
        return srt
        
        if current:
            merged_segments.append(current)

        # 格式化输出
        if output_format == "srt":
            return self._format_as_srt(merged_segments)
        else:
            return "\n".join([s["text"].strip() for s in merged_segments])

    def _format_as_srt(self, segments: List[Dict]) -> str:
        """解析 JSON 偏移量并格式化为 SRT"""
        srt = ""
        for i, seg in enumerate(segments):
            # 【修正】Whisper.cpp JSON offsets 单位确实是毫秒 (ms)
            start_ms = seg["offsets"]["from"]
            end_ms = seg["offsets"]["to"]
            
            def to_srt_time(total_ms):
                total_ms = int(total_ms)
                h = total_ms // 3600000
                m = (total_ms % 3600000) // 60000
                s = (total_ms % 60000) // 1000
                ms = total_ms % 1000
                return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

            srt += f"{i+1}\n"
            srt += f"{to_srt_time(start_ms)} --> {to_srt_time(end_ms)}\n"
            srt += f"{seg['text'].strip()}\n\n"
        return srt

    def transcribe_to_file(
        self,
        audio_path: str,
        output_path: str,
        output_format: str = "txt",
        verbose: bool = True
    ) -> bool:
        """
        转录音频并保存到指定文件
        
        Args:
            audio_path: 输入音频路径
            output_path: 输出文件路径
            output_format: 输出格式
            verbose: 是否显示详细输出
            
        Returns:
            是否成功
        """
        result = self.transcribe(audio_path, output_format, verbose)
        
        if result['success']:
            # 复制到目标位置
            import shutil
            shutil.copy2(result['output_file'], output_path)
            
            # 清理所有相关的临时文件 (json, srt, txt 等)
            import glob
            output_base = result['output_file'].rsplit('.', 1)[0]
            for f in glob.glob(f"{output_base}*"):
                try:
                    os.unlink(f)
                except:
                    pass
            
            if verbose:
                print(f"💾 已保存到: {output_path}")
            return True
        
        return False
    
    def transcribe_to_desktop(
        self,
        audio_path: str,
        output_format: str = "srt"
    ) -> str:
        """
        性能优化版：一次转录，同时保存 SRT 和 TXT
        """
        # 准备输出路径
        audio_name = Path(audio_path).stem
        
        # 创建管理的输出文件夹 (在 Downloads 下)
        downloads = Path.home() / "Downloads"
        output_root = downloads / f"{audio_name}_output"
        output_root.mkdir(parents=True, exist_ok=True)
        
        # 只运行一次转录任务，获取核心数据
        # 默认请求 srt 格式，内部会生成 JSON 并进行智能处理
        result = self.transcribe(audio_path, output_format="srt", verbose=True)
        
        if not result['success']:
            return None

        # 1. 保存 SRT 文件
        srt_path = output_root / f"{audio_name}.srt"
        with open(srt_path, 'w', encoding='utf-8') as f:
            f.write(result['text'])
            
        # 2. 生成并保存 TXT 文件 (直接从结果中提取纯文本)
        # 逻辑：去除时间戳，只保留文本内容
        txt_path = output_root / f"{audio_name}.txt"
        lines = result['text'].split('\n')
        pure_text = []
        for line in lines:
            # 过滤掉 SRT 的数字索引和时间轴行
            if line.strip() and not line.strip().isdigit() and '-->' not in line:
                pure_text.append(line.strip())
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(" ".join(pure_text))
            
        # 3. 复制 JSON 文件过来 (作为备份和数据源)
        if result.get('output_file'):
            src_json = Path(result['output_file']).with_suffix('.json')
            if src_json.exists():
                import shutil
                dst_json = output_root / f"{audio_name}.json"
                try:
                    shutil.copy2(src_json, dst_json)
                except Exception as e:
                    print(f"⚠️ 无法复制 JSON: {e}")

        if self.model_name == "large-v3":
            print(f"✨ M4 性能全开优化：一次运行已同时生成 SRT 和 TXT")
        
        print(f"💾 所有文件已归档至文件夹: {output_root}")
                    
        return result['text']


# 便捷函数
def transcribe_audio(
    audio_path: str,
    model: str = "small",
    language: str = "en",
    output_path: Optional[str] = None
) -> str:
    """
    便捷的转录函数
    
    Args:
        audio_path: 音频文件路径
        model: 模型大小 (tiny, base, small, medium, large)
        language: 语言代码
        output_path: 输出路径（可选，默认保存到 Downloads 文件夹，生成 .srt 和 .txt）
        
    Returns:
        转录文本 (SRT 格式)
    """
    whisper = WhisperCPP(model_name=model, language=language)
    
    if output_path:
        # 如果指定了路径，我们仍然默认生成 srt
        if not output_path.endswith('.srt') and not output_path.endswith('.txt'):
            output_path += ".srt"
        whisper.transcribe_to_file(audio_path, output_path)
        with open(output_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        return whisper.transcribe_to_desktop(audio_path)


# 示例使用
if __name__ == "__main__":
    # 方式 1: 使用类
    print("=" * 60)
    print("方式 1: 使用 WhisperCPP 类")
    print("=" * 60)
    
    whisper = WhisperCPP(model_name="medium", threads=8)
    audio_file = "/Users/bogeling/Downloads/This game theory problem will change the way you see the world.mp4"
    
    # 转录到桌面
    text = whisper.transcribe_to_desktop(audio_file)
    print(f"\n📝 转录预览:\n{text[:500]}...\n")
    
    # 方式 2: 使用便捷函数
    print("=" * 60)
    print("方式 2: 使用便捷函数")
    print("=" * 60)
    
    # text = transcribe_audio(audio_file, model="small")
    # print(f"\n📝 转录完成！")
