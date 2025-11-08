import os
import torch
from faster_whisper import WhisperModel
from transformers import MarianMTModel, MarianTokenizer, AutoTokenizer, AutoModelForSeq2SeqLM
from huggingface_hub import HfApi, model_info
import re
import time
from collections import defaultdict
import json
import subprocess
import sys

def download_faster_whisper_models():
    """下载 faster-whisper 的所有官方标准尺寸模型"""
    print("--- 开始下载 faster-whisper 模型 ---")
    
    # faster-whisper 支持的模型列表
    faster_whisper_model_names = [
        "tiny", "tiny.en",
        "base", "base.en",
        "small", "small.en",
        "medium", "medium.en",
        "large", "large-v2", "large-v3"
    ]
    
    # 确定设备和计算类型
    if torch.cuda.is_available():
        device = "cuda"
        compute_type = "float16"
        print(f"检测到 CUDA 可用，将使用 device='{device}', compute_type='{compute_type}' 加载模型以触发下载。")
    else:
        device = "cpu"
        compute_type = "float32"
        print(f"CUDA 不可用，将回退到 device='{device}', compute_type='{compute_type}'。")

    success_count = 0
    total_count = len(faster_whisper_model_names)
    
    for model_name in faster_whisper_model_names:
        print(f"正在下载/加载 faster-whisper '{model_name}' 模型 (设备: {device}, 精度: {compute_type})...")
        try:
            model = WhisperModel(model_name, device=device, compute_type=compute_type)
            print(f"   [成功] faster-whisper '{model_name}' 模型下载/验证完成。\n")
            success_count += 1
            del model
        except Exception as e:
            print(f"   [失败] 下载/加载 faster-whisper '{model_name}' 模型时出错: {e}\n")
            
    print(f"--- faster-whisper 模型下载完成 ({success_count}/{total_count}) ---\n")


def get_all_helsinki_models():
    """获取Helsinki-NLP的所有翻译模型（排除圣经相关模型）"""
    print("正在从Hugging Face获取Helsinki-NLP的所有模型...")
    
    # 定义目标语言代码（包括可能的变体）
    target_languages = {
        'zh': ['zh', 'zhs', 'zht', 'zh-CN', 'zh-TW', 'zhx'],  # 中文各种变体
        'en': ['en'],  # 英语
        'ja': ['ja'],  # 日语
        'fr': ['fr'],  # 法语
        'de': ['de'],  # 德语
        'ru': ['ru'],  # 俄语
        'ko': ['ko'],  # 韩语
        'es': ['es'],  # 西班牙语
        'it': ['it'],  # 意大利语
        'pt': ['pt'],  # 葡萄牙语
        'ar': ['ar'],  # 阿拉伯语
    }
    
    # 创建所有目标语言变体的扁平列表，便于检查
    all_target_variants = []
    for lang_variants in target_languages.values():
        all_target_variants.extend(lang_variants)
    
    # 初始化Hugging Face API
    api = HfApi()
    
    # 获取所有Helsinki-NLP模型
    all_models = []
    try:
        # 分批获取，避免一次性获取太多
        print("正在获取Helsinki-NLP模型列表...")
        models_iter = api.list_models(author="Helsinki-NLP", limit=1000)
        all_models = list(models_iter)
        print(f"获取到 {len(all_models)} 个Helsinki-NLP模型")
    except Exception as e:
        print(f"获取模型列表失败: {e}")
        return []
    
    # 筛选出翻译模型（排除圣经相关模型）
    translation_models = []
    
    print("\n筛选翻译模型（排除圣经相关模型）...")
    for model in all_models:
        model_id = model.modelId
        
        # 跳过圣经相关模型
        if 'bible' in model_id.lower():
            continue
            
        # 检查是否是翻译模型并提取语言对
        # 匹配多种格式:
        # 1. Helsinki-NLP/opus-mt-src-lang-tgt-lang
        # 2. Helsinki-NLP/opus-mt-tc-big-lang_group-tgt-lang
        # 3. Helsinki-NLP/opus-tatoeba-src-lang-tgt-lang
        
        # 使用更灵活的正则表达式匹配多种格式
        patterns = [
            r'Helsinki-NLP/opus-mt-([a-zA-Z_]+)-([a-zA-Z_]+)$',  # 简单格式: opus-mt-src-tgt
            r'Helsinki-NLP/opus-mt-([a-zA-Z_]+)-([a-zA-Z_]+)-([a-zA-Z_]+)$',  # 中间一个部分: opus-mt-mid-src-tgt
            r'Helsinki-NLP/opus-mt-([a-zA-Z_]+)-([a-zA-Z_]+)-([a-zA-Z_]+)-([a-zA-Z_]+)$',  # 中间两个部分: opus-mt-mid1-mid2-src-tgt
            r'Helsinki-NLP/opus-tatoeba-([a-zA-Z_]+)-([a-zA-Z_]+)$',  # tatoeba格式: opus-tatoeba-src-tgt
        ]
        
        src_lang = None
        tgt_lang = None
        
        for pattern in patterns:
            match = re.search(pattern, model_id)
            if match:
                groups = match.groups()
                if len(groups) == 2:  # 简单格式或tatoeba格式
                    src_lang = groups[0]
                    tgt_lang = groups[1]
                elif len(groups) == 3:  # 中间一个部分
                    src_lang = groups[1]
                    tgt_lang = groups[2]
                elif len(groups) == 4:  # 中间两个部分
                    src_lang = groups[2]
                    tgt_lang = groups[3]
                break
        
        # 如果成功提取了语言对
        if src_lang and tgt_lang:
            # 处理语言组（如 ces_slk）
            src_langs = src_lang.split('_')
            tgt_langs = tgt_lang.split('_')
            
            # 检查源语言和目标语言是否都在目标语言列表中
            src_is_target = any(lang in all_target_variants for lang in src_langs)
            tgt_is_target = any(lang in all_target_variants for lang in tgt_langs)
            
            if src_is_target and tgt_is_target:
                translation_models.append(model_id)
                print(f"  [包含] {model_id} ({src_lang} -> {tgt_lang})")
    
    print(f"筛选出 {len(translation_models)} 个目标语言之间的互译模型")
    return translation_models


def get_target_nllb_models():
    """获取NLLB模型"""
    print("正在检索NLLB模型...")
    
    # 使用预定义的NLLB模型列表（这些是确认存在的）
    nllb_models = [
        "facebook/nllb-200-distilled-600M",
        "facebook/nllb-200-distilled-1.3B",
        "facebook/nllb-200-1.3B",
        "facebook/nllb-200-3.3B",
    ]
    
    # 验证模型是否存在
    valid_models = []
    for model_id in nllb_models:
        try:
            # 检查模型是否存在
            info = model_info(model_id)
            if info:
                valid_models.append(model_id)
                print(f"  [有效] {model_id}")
        except Exception as e:
            print(f"  [无效] {model_id}")
            continue
    
    print(f"验证后找到 {len(valid_models)} 个有效的NLLB模型")
    return valid_models


def extract_language_pair(model_id):
    """从模型ID中提取语言对"""
    # 匹配多种格式的正则表达式
    patterns = [
        r'Helsinki-NLP/opus-mt-([a-zA-Z_]+)-([a-zA-Z_]+)$',  # 简单格式: opus-mt-src-tgt
        r'Helsinki-NLP/opus-mt-([a-zA-Z_]+)-([a-zA-Z_]+)-([a-zA-Z_]+)$',  # 中间一个部分: opus-mt-mid-src-tgt
        r'Helsinki-NLP/opus-mt-([a-zA-Z_]+)-([a-zA-Z_]+)-([a-zA-Z_]+)-([a-zA-Z_]+)$',  # 中间两个部分: opus-mt-mid1-mid2-src-tgt
        r'Helsinki-NLP/opus-tatoeba-([a-zA-Z_]+)-([a-zA-Z_]+)$',  # tatoeba格式: opus-tatoeba-src-tgt
    ]
    
    for pattern in patterns:
        match = re.search(pattern, model_id)
        if match:
            groups = match.groups()
            if len(groups) == 2:  # 简单格式或tatoeba格式
                src_lang = groups[0]
                tgt_lang = groups[1]
            elif len(groups) == 3:  # 中间一个部分
                src_lang = groups[1]
                tgt_lang = groups[2]
            elif len(groups) == 4:  # 中间两个部分
                src_lang = groups[2]
                tgt_lang = groups[3]
            return f"{src_lang}->{tgt_lang}"
    
    # NLLB模型支持所有语言对
    if "nllb" in model_id.lower():
        return "多语言->多语言"
    
    return None


def download_translation_models():
    """下载目标语言之间的互译模型"""
    print("--- 开始下载 transformers 翻译模型 ---")
    
    # 获取所有Helsinki-NLP翻译模型
    helsinki_models = get_all_helsinki_models()
    
    # 获取NLLB模型
    nllb_models = get_target_nllb_models()
    
    # 合并所有模型
    translation_models_to_download = helsinki_models + nllb_models
    
    success_count = 0
    total_count = len(translation_models_to_download)
    
    # 记录成功下载的模型
    successful_models = []
    
    print(f"\n计划下载 {total_count} 个翻译模型...")
    print("包含以下模型:")
    for model_id in translation_models_to_download:
        print(f"  - {model_id}")
    print()
    
    # 检查CUDA可用性（仅用于显示信息）
    if torch.cuda.is_available():
        print("检测到 CUDA 可用，但模型将仅下载而不加载到 GPU。")

    for model_name in translation_models_to_download:
        print(f"正在下载 translation model '{model_name}'...")
        try:
            # 根据模型类型选择正确的tokenizer和模型加载方式
            if "nllb" in model_name.lower():
                # NLLB模型使用AutoTokenizer和AutoModelForSeq2SeqLM
                # 仅下载而不加载到GPU
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            else:
                # Helsinki-NLP模型使用MarianTokenizer和MarianMTModel
                # 仅下载而不加载到GPU
                tokenizer = MarianTokenizer.from_pretrained(model_name)
                model = MarianMTModel.from_pretrained(model_name)
                 
            print(f"   [成功] '{model_name}' 模型下载/验证完成。\n")
            success_count += 1
            successful_models.append(model_name)
            del model, tokenizer
            
            # 添加短暂延迟，避免请求过于频繁
            time.sleep(0.5)
        except Exception as e:
            print(f"   [失败] 下载 '{model_name}' 模型时出错: {e}\n")
            
    print(f"--- transformers 翻译模型下载完成 ({success_count}/{total_count}) ---\n")
    
    # 返回成功下载的模型列表
    return successful_models


def create_language_model_dict(successful_models):
    """创建语言对到模型的字典"""
    language_model_dict = defaultdict(list)
    
    for model_id in successful_models:
        lang_pair = extract_language_pair(model_id)
        if lang_pair:
            language_model_dict[lang_pair].append(model_id)
    
    # 转换为普通字典
    return dict(language_model_dict)


def print_language_model_dict(language_model_dict):
    """打印语言对模型字典"""
    print("\n" + "=" * 60)
    print("可用语言对及对应模型字典")
    print("=" * 60)
    
    # 按语言对排序
    sorted_pairs = sorted(language_model_dict.keys())
    
    for lang_pair in sorted_pairs:
        models = language_model_dict[lang_pair]
        print(f"\n[{lang_pair}]")
        for model in models:
            print(f"  - {model}")
    
    print("\n" + "=" * 60)
    print("字典格式输出（可用于程序直接使用）")
    print("=" * 60)
    print("{")
    for i, lang_pair in enumerate(sorted_pairs):
        models = language_model_dict[lang_pair]
        print(f"    '{lang_pair}': [")
        for j, model in enumerate(models):
            comma = "," if j < len(models) - 1 else ""
            print(f"        '{model}'{comma}")
        closing_bracket = "}," if i < len(sorted_pairs) - 1 else "}"
        print(f"    ]{closing_bracket}")
    print("}")
    
    # 保存到文件
    try:
        with open("language_model_dict.json", "w", encoding="utf-8") as f:
            json.dump(language_model_dict, f, ensure_ascii=False, indent=2)
        print("\n字典已保存到 language_model_dict.json 文件")
    except Exception as e:
        print(f"\n保存字典到文件失败: {e}")


def get_kokoro_models():
    """获取Kokoro TTS模型列表"""
    print("正在获取Kokoro TTS模型列表...")
    
    # Kokoro TTS模型列表（基于hexgrad/kokoro仓库）
    kokoro_models = {
        'en': ['af_heart', 'af_sky', 'am_adam', 'am_michael'],  # 英语语音
        'zh': ['zf_xiaoya', 'zf_xiaobei', 'zm_xuan'],  # 中文语音
        'ja': ['jf_alpha_0', 'jf_kumo_0', 'jf_gongitsune_0', 'jf_nezumi_0', 'jf_usagi_0'],  # 日语语音
    }
    
    # 对于不支持的语言，使用英语语音作为后备
    fallback_langs = ['es', 'fr', 'de', 'ru', 'ko', 'it', 'pt', 'ar']
    for lang in fallback_langs:
        kokoro_models[lang] = kokoro_models['en']
    
    return kokoro_models


def download_kokoro_models():
    """下载Kokoro TTS模型"""
    print("--- 开始下载 Kokoro TTS 模型 ---")
    
    # 首先检查是否安装了kokoro
    try:
        import importlib
        kokoro_spec = importlib.util.find_spec("kokoro")
        if kokoro_spec is None:
            print("Kokoro未安装，正在尝试安装...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "kokoro"])
            print("Kokoro安装完成")
    except Exception as e:
        print(f"安装Kokoro时出错: {e}")
        print("请手动安装Kokoro: pip install kokoro")
        return {}
    
    # 获取Kokoro模型列表
    kokoro_models = get_kokoro_models()
    
    # 记录成功下载的模型
    successful_models = {}
    
    # 尝试导入Kokoro
    try:
        from kokoro import KPipeline
        print("成功导入Kokoro")
    except ImportError as e:
        print(f"导入Kokoro失败: {e}")
        return {}
    
    # 尝试下载每种语言的模型
    for lang, voices in kokoro_models.items():
        print(f"\n正在下载 {lang} 语言的 Kokoro 模型...")
        
        # 确定语言代码
        lang_code = 'a'  # 默认英语
        if lang == 'zh':
            lang_code = 'z'
        elif lang == 'ja':
            lang_code = 'j'
        
        successful_voices = []
        
        # 尝试初始化KPipeline以触发模型下载
        try:
            pipeline = KPipeline(lang_code=lang_code)
            print(f"   [成功] {lang} 语言模型下载/验证完成")
            successful_voices = voices
            successful_models[lang] = voices
            del pipeline
        except Exception as e:
            print(f"   [失败] 下载 {lang} 语言模型时出错: {e}")
        
        # 尝试每个语音模型
        for voice in voices:
            print(f"   正在验证语音模型: {voice}")
            try:
                # 尝试使用语音模型生成一个短音频片段
                pipeline = KPipeline(lang_code=lang_code)
                generator = pipeline("Hello", voice=voice)
                # 获取第一个结果（不实际使用，只是为了触发下载）
                next(generator)
                print(f"      [成功] 语音模型 {voice} 验证完成")
                del pipeline, generator
            except Exception as e:
                print(f"      [失败] 验证语音模型 {voice} 时出错: {e}")
    
    # 保存Kokoro模型字典
    try:
        with open("kokoro_model_dict.json", "w", encoding="utf-8") as f:
            json.dump(successful_models, f, ensure_ascii=False, indent=2)
        print("\nKokoro模型字典已保存到 kokoro_model_dict.json 文件")
    except Exception as e:
        print(f"\n保存Kokoro模型字典到文件失败: {e}")
    
    print(f"\n--- Kokoro TTS 模型下载完成 ---")
    return successful_models


def get_hf_cache_size():
    """
    估算 Hugging Face 缓存目录的大小。
    """
    hf_cache_dir = os.path.expanduser("~/.cache/huggingface/hub/")
    total_size_bytes = 0
    
    if os.path.exists(hf_cache_dir):
        try:
            for dirpath, dirnames, filenames in os.walk(hf_cache_dir):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    total_size_bytes += os.path.getsize(fp)
        except Exception as e:
            print(f"计算 Hugging Face 缓存大小时发生错误: {e}")
            return None
    else:
        print(f"Hugging Face 缓存目录不存在: {hf_cache_dir}")
        return 0
        
    size_gb = total_size_bytes / (1024**3)
    return size_gb


def main():
    """
    主函数：执行所有模型的预下载。
    """
    print("=" * 60)
    print("开始预下载所有指定的 faster-whisper、翻译模型和Kokoro TTS模型文件...")
    print("=" * 60)
    print("注意：这将下载大量模型文件，请确保网络和磁盘空间充足。")
    print("=" * 60)
    print("目标语言：中文、英语、日语、法语、德语、俄语、韩语、西班牙语、意大利语、葡萄牙语、阿拉伯语")
    print("=" * 60)
    
    # 1. 下载 faster-whisper 模型
    download_faster_whisper_models()
    
    # 2. 下载 translation models
    successful_models = download_translation_models()
    
    # 3. 创建并打印语言对模型字典
    if successful_models:
        language_model_dict = create_language_model_dict(successful_models)
        print_language_model_dict(language_model_dict)
    else:
        print("\n没有成功下载任何翻译模型。")
    
    # 4. 下载 Kokoro TTS 模型
    kokoro_models = download_kokoro_models()
    
    # 5. 显示缓存占用空间
    print("\n--- 缓存空间估算 ---")
    cache_size_gb = get_hf_cache_size()
    if cache_size_gb is not None:
        print(f"估算的 Hugging Face 模型缓存总大小: {cache_size_gb:.2f} GB")
    print("-" * 60)

    print("\n所有模型预下载任务已完成。")

if __name__ == "__main__":
    # 添加一个简单的确认步骤，防止意外运行
    # user_input = input("即将开始下载大量模型文件 (~50GB+)。确认继续吗？(输入 'yes' 继续): ")
    # if user_input.lower() != 'yes':
    #     print("操作已取消。")
    #     exit(0)
        
    main()