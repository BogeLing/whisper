"""
Whisper 转录使用示例 - 修改这个文件中的路径来使用
"""

from whisper_cpp_wrapper import transcribe_audio, WhisperCPP

# ============================================
# 快速开始：修改下面的路径然后运行
# ============================================
if __name__ == "__main__":
    print("=" * 60)
    print("Whisper 音频转录工具")
    print("=" * 60)
    print("📥 默认保存位置：~/Downloads/")
    print("📝 默认生成：.srt (字幕) 和 .txt (文本) 文件")
    print("=" * 60)
    
    # 🔧 修改这里：你的音频文件路径
    input_audio = "/Users/bogeling/Downloads/Building_Scalable_Game_Engines_From_Scratch.m4a"
    
    # 方式 A：使用默认设置（自动保存到 Downloads）
    text = transcribe_audio(
        audio_path=input_audio,
        model="large-v3",  # 使用 large-v3 模型
        language="en"    # 语言设置为英语
    )
    
    print(f"\n✅ 转录完成！")
    print(f"📁 已自动保存到 ~/Downloads/ 下的 .srt 和 .txt 文件")
    print(f"📝 字幕预览:\n{text[:300]}...")
    
    # 方式 B：如果要自定义输出路径，使用这个：
    # output_text = "/Users/bogeling/Documents/我的转录.txt"
    # text = transcribe_audio(input_audio, output_path=output_text, model="medium")
    # print(f"📁 已保存到: {output_text}")
