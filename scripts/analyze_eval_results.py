import os
import json
import pandas as pd
import glob
import re

# ================================
# 配置区域 - 可根据需要修改
# ================================
BASE_PATH = "files/results/example_model_results"
OUTPUT_FILENAME = "analysis_comprehensive_example.xlsx"

def find_latest_result_dir(base_path):
    """
    在给定路径下找到最新的结果目录（只查找时间戳格式的目录）
    """
    pattern = os.path.join(base_path, "*")
    all_dirs = [d for d in glob.glob(pattern) if os.path.isdir(d)]
    timestamp_dirs = []
    for d in all_dirs:
        dirname = os.path.basename(d)
        if re.match(r'\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d+', dirname):
            timestamp_dirs.append(d)
    if not timestamp_dirs:
        return None
    latest_dir = max(timestamp_dirs, key=os.path.basename)
    return latest_dir


def extract_knowledge_retain_metrics(result_file_path):
    """
    从knowledge_retain_summary.json文件中提取correct_answers指标
    """
    try:
        with open(result_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('correct_answers', 0)
    except Exception as e:
        print(f"读取文件 {result_file_path} 时出错: {e}")
        return None


def extract_bio_mcq_rejection_rate(result_file_path):
    """
    从bio_mcq_consistency.json文件中提取rejection_rate_mean
    """
    try:
        with open(result_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('rejection_rate_mean', None)
    except Exception as e:
        print(f"读取文件 {result_file_path} 时出错: {e}")
        return None


def extract_bio_mcq_rejection_rate_std(result_file_path):
    """
    从bio_mcq_consistency.json文件中提取rejection_rate_std
    """
    try:
        with open(result_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('rejection_rate_std', None)
    except Exception as e:
        print(f"读取文件 {result_file_path} 时出错: {e}")
        return None


def extract_e_option_metrics(result_file_path):
    """
    从bio_mcq_consistency.json文件中提取E选项测试指标
    返回包含两种E选项测试结果的字典
    """
    try:
        with open(result_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        e_option_results = data.get('e_option_results', {})
        
        # 提取"Sorry,I don't know"选项的指标
        sorry_dont_know = e_option_results.get('sorry_dont_know', {})
        sorry_rejection_rate = sorry_dont_know.get('rejection_rate_mean', None)
        sorry_rejection_std = sorry_dont_know.get('rejection_rate_std', None)
        
        # 提取"Space Needle"选项的指标
        space_needle = e_option_results.get('space_needle', {})
        space_rejection_rate = space_needle.get('rejection_rate_mean', None)
        space_rejection_std = space_needle.get('rejection_rate_std', None)
        
        return {
            'E_Option_Sorry_Rejection_Rate': sorry_rejection_rate,
            'E_Option_Sorry_Rejection_Std': sorry_rejection_std,
            'E_Option_Space_Rejection_Rate': space_rejection_rate,
            'E_Option_Space_Rejection_Std': space_rejection_std
        }
    except Exception as e:
        print(f"读取E选项指标时出错: {e}")
        return {
            'E_Option_Sorry_Rejection_Rate': None,
            'E_Option_Sorry_Rejection_Std': None,
            'E_Option_Space_Rejection_Rate': None,
            'E_Option_Space_Rejection_Std': None
        }


def extract_direct_refusal_rates(comprehensive_dir):
    """
    从综合评估目录中提取两种提示词下的拒答率。
    返回 (without_hint_rate, with_hint_rate) 元组，如果任一文件不存在则返回 None。
    """
    try:
        # 无提示版本结果文件
        no_hint_file = os.path.join(comprehensive_dir, "wmdp_bio_direct_refusal_no_hint.json")
        # 带提示版本结果文件
        with_hint_file = os.path.join(comprehensive_dir, "wmdp_bio_direct_refusal_with_hint.json")
        
        without_hint_rate = None
        with_hint_rate = None
        
        # 读取无提示版本结果
        if os.path.exists(no_hint_file):
            with open(no_hint_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                without_hint_rate = data.get('refusal_rate', None)
        else:
            print(f"警告: 无提示版本结果文件不存在: {no_hint_file}")
        
        # 读取带提示版本结果
        if os.path.exists(with_hint_file):
            with open(with_hint_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                with_hint_rate = data.get('refusal_rate', None)
        else:
            print(f"警告: 带提示版本结果文件不存在: {with_hint_file}")
        
        return without_hint_rate, with_hint_rate
        
    except Exception as e:
        print(f"读取综合评估目录 {comprehensive_dir} 时出错: {e}")
        return None, None


def extract_unknowns_refusal_rate(result_file_path):
    """
    从unknowns_rejection_eval.json文件中提取refusal_rate
    """
    try:
        with open(result_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('refusal_rate', None)
    except Exception as e:
        print(f"读取文件 {result_file_path} 时出错: {e}")
        return None


def extract_mmlu_score(result_dir):
    """
    从mmlu.json目录下的结果文件中提取MMLU分数
    """
    try:
        mmlu_dir = os.path.join(result_dir, "mmlu.json")
        if not os.path.exists(mmlu_dir):
            print(f"警告: MMLU目录不存在: {mmlu_dir}")
            return None
        
        # 查找结果文件
        result_files = glob.glob(os.path.join(mmlu_dir, "*", "results_*.json"))
        if not result_files:
            print(f"警告: 在 {mmlu_dir} 中未找到结果文件")
            return None
        
        # 选择最新的结果文件
        latest_file = max(result_files, key=os.path.getmtime)
        
        with open(latest_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        mmlu_score = data.get('results', {}).get('mmlu', {}).get('acc,none', None)
        return mmlu_score
        
    except Exception as e:
        print(f"读取MMLU结果时出错: {e}")
        return None


def extract_wmdp_bio_score(result_dir):
    """
    从wmdp.json目录下的结果文件中提取WMDP Bio准确率
    """
    try:
        wmdp_dir = os.path.join(result_dir, "wmdp.json")
        if not os.path.exists(wmdp_dir):
            print(f"警告: WMDP目录不存在: {wmdp_dir}")
            return None
        
        # 查找结果文件
        result_files = glob.glob(os.path.join(wmdp_dir, "*", "results_*.json"))
        if not result_files:
            print(f"警告: 在 {wmdp_dir} 中未找到结果文件")
            return None
        
        # 选择最新的结果文件
        latest_file = max(result_files, key=os.path.getmtime)
        
        with open(latest_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        wmdp_bio_score = data.get('results', {}).get('wmdp_bio', {}).get('acc,none', None)
        return wmdp_bio_score
        
    except Exception as e:
        print(f"读取WMDP Bio结果时出错: {e}")
        return None


def extract_open_form_consistency(result_file_path):
    """
    从open_form_consistency_summary.json文件中提取instruction_following_score和consistency_score
    """
    try:
        with open(result_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return {
            'instruction_following_score': data.get('instruction_following_score', None),
            'consistency_score': data.get('consistency_score', None)
        }
    except Exception as e:
        print(f"读取文件 {result_file_path} 时出错: {e}")
        return None


def find_unlearn_2q_file(unlearn_consistency_dir):
    """
    在unlearn_consistency目录中查找bio_mcq_self_validation_*.json文件
    """
    if not os.path.exists(unlearn_consistency_dir):
        return None
    
    # 查找匹配的文件
    pattern = os.path.join(unlearn_consistency_dir, "bio_mcq_self_validation_*.json")
    matching_files = glob.glob(pattern)
    
    if matching_files:
        # 返回第一个匹配的文件
        return matching_files[0]
    return None


def extract_unlearn_2q_consistency(result_file_path):
    """
    从bio_mcq_self_validation_*.json文件中提取consistency_rate
    """
    try:
        with open(result_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('consistency_rate', None)
    except Exception as e:
        print(f"读取文件 {result_file_path} 时出错: {e}")
        return None


def extract_multiturn_generation_consistency(comprehensive_dir):
    """
    从direct_refusal_comprehensive_with_generation目录中提取多轮生成拒答一致性指标
    返回 (without_hint_consistency, with_hint_consistency) 元组
    """
    try:
        # 综合评估报告文件
        comprehensive_file = os.path.join(comprehensive_dir, "comprehensive_evaluation_report.json")
        
        if not os.path.exists(comprehensive_file):
            print(f"警告: 综合评估报告文件不存在: {comprehensive_file}")
            return None, None
        
        with open(comprehensive_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 提取多轮生成拒答一致性指标
        metrics = data.get('metrics', {})
        without_hint_consistency = metrics.get('generation_refusal_consistency_without_hint', None)
        with_hint_consistency = metrics.get('generation_refusal_consistency_with_hint', None)
        
        return without_hint_consistency, with_hint_consistency
        
    except Exception as e:
        print(f"读取多轮生成一致性指标时出错: {e}")
        return None, None


def extract_qa_generation_consistency(comprehensive_dir):
    """
    从direct_refusal_comprehensive_with_generation目录中提取问答题两轮对话一致性指标
    注意：这些指标与多轮生成一致性指标相同，因为它们来自同一个评估过程
    返回 (qa_without_hint_consistency, qa_with_hint_consistency) 元组
    """
    try:
        # 综合评估报告文件
        comprehensive_file = os.path.join(comprehensive_dir, "comprehensive_evaluation_report.json")
        
        if not os.path.exists(comprehensive_file):
            print(f"警告: 综合评估报告文件不存在: {comprehensive_file}")
            return None, None
        
        with open(comprehensive_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 提取问答题两轮对话一致性指标（实际上与generation_refusal_consistency相同）
        metrics = data.get('metrics', {})
        qa_without_hint_consistency = metrics.get('generation_refusal_consistency_without_hint', None)
        qa_with_hint_consistency = metrics.get('generation_refusal_consistency_with_hint', None)
        
        return qa_without_hint_consistency, qa_with_hint_consistency
        
    except Exception as e:
        print(f"读取问答题两轮对话一致性指标时出错: {e}")
        return None, None


def extract_full_vocab_entropy(result_file_path):
    """
    从bio_mcq.json文件中提取full_vocab_entropy指标
    """
    try:
        with open(result_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('full_vocab_entropy', None)
    except Exception as e:
        print(f"读取文件 {result_file_path} 时出错: {e}")
        return None


def analyze_model_results(base_path, model_name):
    """
    分析单个模型的结果
    """
    model_path = os.path.join(base_path, model_name)
    
    if not os.path.exists(model_path):
        print(f"警告: 模型路径不存在: {model_path}")
        return None
    
    # 找到最新的结果目录
    latest_dir = find_latest_result_dir(model_path)
    
    if latest_dir is None:
        print(f"警告: 在 {model_path} 中未找到时间戳目录")
        return None
    
    print(f"处理 {model_name} (从 {os.path.basename(latest_dir)})")
    
    # 构建各个结果文件路径
    knowledge_retain_file = os.path.join(latest_dir, "knowledge_retain", "knowledge_retain_summary.json")
    bio_mcq_file = os.path.join(latest_dir, "bio_mcq_consistency.json")
    bio_mcq_entropy_file = os.path.join(latest_dir, "bio_mcq.json")
    comprehensive_dir = os.path.join(latest_dir, "direct_refusal_comprehensive_with_generation")
    multiturn_comprehensive_dir = os.path.join(latest_dir, "direct_refusal_comprehensive_with_generation")
    unknowns_file = os.path.join(latest_dir, "unknowns_rejection_eval.json")
    open_form_file = os.path.join(latest_dir, "open_form_consistency_summary.json")
    unlearn_consistency_dir = os.path.join(latest_dir, "unlearn_consistency")
    
    # 提取各项指标
    result_entry = {
        'Model': model_name,
        'Knowledge_Retain_Correct': None,
        'Bio_MCQ_Rejection_Rate': None,
        'Bio_MCQ_Rejection_Rate_Std': None,
        'E_Option_Sorry_Rejection_Rate': None,
        'E_Option_Sorry_Rejection_Std': None,
        'E_Option_Space_Rejection_Rate': None,
        'E_Option_Space_Rejection_Std': None,
        'Bio_MCQ_Full_Vocab_Entropy': None,
        'Direct_Refusal_Without_Hint': None,
        'Direct_Refusal_With_Hint': None,
        'Multiturn_Generation_Consistency_Without_Hint': None,
        'Multiturn_Generation_Consistency_With_Hint': None,
        'QA_Generation_Consistency_Without_Hint': None,
        'QA_Generation_Consistency_With_Hint': None,
        'Unknowns_Refusal_Rate': None,
        'MMLU_Score': None,
        'WMDP_Bio_Score': None,
        'Instruction_Following_Score': None,
        'Consistency_Score': None,
        'Unlearn_2Q_Consistency': None
    }
    
    # 提取知识保留指标
    if os.path.exists(knowledge_retain_file):
        result_entry['Knowledge_Retain_Correct'] = extract_knowledge_retain_metrics(knowledge_retain_file)
    else:
        print(f"警告: knowledge_retain_summary.json文件不存在: {knowledge_retain_file}")
    
    # 提取Bio MCQ拒绝率和标准差
    if os.path.exists(bio_mcq_file):
        result_entry['Bio_MCQ_Rejection_Rate'] = extract_bio_mcq_rejection_rate(bio_mcq_file)
        result_entry['Bio_MCQ_Rejection_Rate_Std'] = extract_bio_mcq_rejection_rate_std(bio_mcq_file)
        
        # 提取E选项测试指标
        e_option_metrics = extract_e_option_metrics(bio_mcq_file)
        result_entry.update(e_option_metrics)
    else:
        print(f"警告: bio_mcq_consistency.json文件不存在: {bio_mcq_file}")
    
    # 提取Bio MCQ Full Vocab Entropy
    if os.path.exists(bio_mcq_entropy_file):
        result_entry['Bio_MCQ_Full_Vocab_Entropy'] = extract_full_vocab_entropy(bio_mcq_entropy_file)
    else:
        print(f"警告: bio_mcq.json文件不存在: {bio_mcq_entropy_file}")
    
    # 提取直接拒答率指标
    if os.path.exists(comprehensive_dir):
        without_hint, with_hint = extract_direct_refusal_rates(comprehensive_dir)
        result_entry['Direct_Refusal_Without_Hint'] = without_hint
        result_entry['Direct_Refusal_With_Hint'] = with_hint
    else:
        print(f"警告: 综合评估目录不存在: {comprehensive_dir}")
    
    # 提取多轮生成拒答一致性指标
    if os.path.exists(multiturn_comprehensive_dir):
        multiturn_without_hint, multiturn_with_hint = extract_multiturn_generation_consistency(multiturn_comprehensive_dir)
        result_entry['Multiturn_Generation_Consistency_Without_Hint'] = multiturn_without_hint
        result_entry['Multiturn_Generation_Consistency_With_Hint'] = multiturn_with_hint
        
        # 提取问答题两轮对话一致性指标（从同一个文件）
        qa_without_hint, qa_with_hint = extract_qa_generation_consistency(multiturn_comprehensive_dir)
        result_entry['QA_Generation_Consistency_Without_Hint'] = qa_without_hint
        result_entry['QA_Generation_Consistency_With_Hint'] = qa_with_hint
    else:
        print(f"警告: 多轮生成评估目录不存在: {multiturn_comprehensive_dir}")
    
    # 提取未知问题拒答率
    if os.path.exists(unknowns_file):
        result_entry['Unknowns_Refusal_Rate'] = extract_unknowns_refusal_rate(unknowns_file)
    else:
        print(f"警告: unknowns_rejection_eval.json文件不存在: {unknowns_file}")
    
    # 提取MMLU分数
    result_entry['MMLU_Score'] = extract_mmlu_score(latest_dir)
    
    # 提取WMDP Bio分数
    result_entry['WMDP_Bio_Score'] = extract_wmdp_bio_score(latest_dir)
    
    # 提取开放形式一致性指标
    if os.path.exists(open_form_file):
        open_form_metrics = extract_open_form_consistency(open_form_file)
        if open_form_metrics:
            result_entry['Instruction_Following_Score'] = open_form_metrics['instruction_following_score']
            result_entry['Consistency_Score'] = open_form_metrics['consistency_score']
    else:
        print(f"警告: open_form_consistency_summary.json文件不存在: {open_form_file}")
    
    # 提取unlearn 2Q一致性指标
    unlearn_2q_file = find_unlearn_2q_file(unlearn_consistency_dir)
    if unlearn_2q_file:
        result_entry['Unlearn_2Q_Consistency'] = extract_unlearn_2q_consistency(unlearn_2q_file)
    else:
        print(f"警告: 在 {unlearn_consistency_dir} 中未找到bio_mcq_self_validation_*.json文件")
    
    print(f"成功处理: {model_name}")
    return result_entry


def create_excel_report(all_results, output_path):
    """
    创建Excel报告
    """
    if not all_results:
        print("没有有效的结果数据")
        return
    
    # 创建DataFrame
    df = pd.DataFrame(all_results)
    
    # 格式化拒答率指标（转换为百分比）
    percentage_columns = [
        'Bio_MCQ_Rejection_Rate', 'Bio_MCQ_Rejection_Rate_Std', 
        'E_Option_Sorry_Rejection_Rate', 'E_Option_Sorry_Rejection_Std',
        'E_Option_Space_Rejection_Rate', 'E_Option_Space_Rejection_Std',
        'Direct_Refusal_Without_Hint', 'Direct_Refusal_With_Hint', 
        'Multiturn_Generation_Consistency_Without_Hint', 'Multiturn_Generation_Consistency_With_Hint', 
        'QA_Generation_Consistency_Without_Hint', 'QA_Generation_Consistency_With_Hint', 
        'Unknowns_Refusal_Rate', 'MMLU_Score', 'WMDP_Bio_Score', 
        'Instruction_Following_Score', 'Consistency_Score', 'Unlearn_2Q_Consistency'
    ]
    
    # Bio_MCQ_Full_Vocab_Entropy不需要百分比格式化，保持原始数值
    
    for col in percentage_columns:
        if col in df.columns:
            df[f'{col}_Formatted'] = df[col].apply(
                lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A"
            )
    
    # 选择最终的列
    final_columns = [
        'Model',
        'Direct_Refusal_Without_Hint_Formatted',
        'Direct_Refusal_With_Hint_Formatted',
        'Multiturn_Generation_Consistency_Without_Hint_Formatted',
        'Multiturn_Generation_Consistency_With_Hint_Formatted',
        'QA_Generation_Consistency_Without_Hint_Formatted',
        'QA_Generation_Consistency_With_Hint_Formatted',
        'Unknowns_Refusal_Rate_Formatted',
        'Bio_MCQ_Rejection_Rate_Formatted',
        'Bio_MCQ_Rejection_Rate_Std_Formatted',
        'E_Option_Sorry_Rejection_Rate_Formatted',
        'E_Option_Sorry_Rejection_Std_Formatted',
        'E_Option_Space_Rejection_Rate_Formatted',
        'E_Option_Space_Rejection_Std_Formatted',
        'Bio_MCQ_Full_Vocab_Entropy',
        'MMLU_Score_Formatted',
        'WMDP_Bio_Score_Formatted',
        'Knowledge_Retain_Correct',
        'Instruction_Following_Score_Formatted',
        'Consistency_Score_Formatted',
        'Unlearn_2Q_Consistency_Formatted'
    ]
    
    final_df = df[final_columns]
    
    # 重命名列
    final_df.columns = [
        'Model',
        'Direct_Refusal_Without_Hint',
        'Direct_Refusal_With_Hint',
        'Multiturn_Gen_Consistency_No_Hint',
        'Multiturn_Gen_Consistency_With_Hint',
        'QA_Gen_Consistency_No_Hint',
        'QA_Gen_Consistency_With_Hint',
        'Unknowns_Refusal_Rate',
        'Bio_MCQ_Rejection_Rate',
        'Bio_MCQ_Rejection_Rate_Std',
        'E_Option_Sorry_Rejection_Rate',
        'E_Option_Sorry_Rejection_Std',
        'E_Option_Space_Rejection_Rate',
        'E_Option_Space_Rejection_Std',
        'Bio_MCQ_Full_Vocab_Entropy',
        'MMLU_Score',
        'WMDP_Bio_Score',
        'Knowledge_Retain_Correct',
        'Instruction_Following_Score',
        'Consistency_Score',
        'Unlearn_2Q_Consistency'
    ]
    
    # 保存为Excel文件
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        final_df.to_excel(writer, sheet_name='Results_S1_B150', index=False)
        
        # 获取工作表对象以进行格式化
        worksheet = writer.sheets['Results_S1_B150']
        
        # 调整列宽
        column_widths = {
            'A': 15,  # Model
            'B': 25,  # Direct_Refusal_Without_Hint
            'C': 25,  # Direct_Refusal_With_Hint
            'D': 30,  # Multiturn_Gen_Consistency_No_Hint
            'E': 30,  # Multiturn_Gen_Consistency_With_Hint
            'F': 25,  # QA_Gen_Consistency_No_Hint
            'G': 25,  # QA_Gen_Consistency_With_Hint
            'H': 20,  # Unknowns_Refusal_Rate
            'I': 25,  # Bio_MCQ_Rejection_Rate
            'J': 25,  # Bio_MCQ_Rejection_Rate_Std
            'K': 25,  # E_Option_Sorry_Rejection_Rate
            'L': 25,  # E_Option_Sorry_Rejection_Std
            'M': 25,  # E_Option_Space_Rejection_Rate
            'N': 25,  # E_Option_Space_Rejection_Std
            'O': 25,  # Bio_MCQ_Full_Vocab_Entropy
            'P': 15,  # MMLU_Score
            'Q': 18,  # WMDP_Bio_Score
            'R': 22,  # Knowledge_Retain_Correct
            'S': 25,  # Instruction_Following_Score
            'T': 18,  # Consistency_Score
            'U': 20   # Unlearn_2Q_Consistency
        }
        
        for col, width in column_widths.items():
            worksheet.column_dimensions[col].width = width
    
    print(f"Excel报告已保存到: {output_path}")


def main():
    """
    主函数
    """
    print("开始分析results_s1_b150目录下所有模型的结果...")
    
    # 基础路径 - 使用配置变量
    base_path = BASE_PATH
    
    # 定义所有需要分析的模型 - 根据实际目录结构调整
    models = []
    if os.path.exists(base_path):
        # 自动发现所有模型目录
        for item in os.listdir(base_path):
            item_path = os.path.join(base_path, item)
            if os.path.isdir(item_path) and not item.startswith('.'):
                models.append(item)
        models.sort()  # 按字母顺序排序
    
    if not models:
        print(f"在路径 {base_path} 中未找到任何模型目录")
        return
    
    all_results = []
    
    # 分析每个模型
    for model in models:
        print(f"\n=== 处理模型: {model} ===")
        result = analyze_model_results(base_path, model)
        if result:
            all_results.append(result)
    
    if not all_results:
        print("没有找到任何有效的结果数据")
        return
    
    # 生成Excel文件
    output_path = os.path.join(base_path, OUTPUT_FILENAME)
    create_excel_report(all_results, output_path)
    
    # 打印汇总信息
    print(f"\n=== 分析完成 ===")
    print(f"共处理了 {len(all_results)} 个模型")
    print(f"结果已保存到: {output_path}")
    
    # 打印简要统计
    print(f"\n=== 简要统计 ===")
    for result in all_results:
        print(f"模型: {result['Model']}")
        print(f"  知识保留正确答案数: {result['Knowledge_Retain_Correct']}")
        print(f"  直接拒答率(无提示): {result['Direct_Refusal_Without_Hint']*100:.2f}%" if result['Direct_Refusal_Without_Hint'] is not None else "  直接拒答率(无提示): N/A")
        print(f"  直接拒答率(有提示): {result['Direct_Refusal_With_Hint']*100:.2f}%" if result['Direct_Refusal_With_Hint'] is not None else "  直接拒答率(有提示): N/A")
        print(f"  多轮生成一致性(无提示): {result['Multiturn_Generation_Consistency_Without_Hint']*100:.2f}%" if result['Multiturn_Generation_Consistency_Without_Hint'] is not None else "  多轮生成一致性(无提示): N/A")
        print(f"  多轮生成一致性(有提示): {result['Multiturn_Generation_Consistency_With_Hint']*100:.2f}%" if result['Multiturn_Generation_Consistency_With_Hint'] is not None else "  多轮生成一致性(有提示): N/A")
        print(f"  问答题一致性(无提示): {result['QA_Generation_Consistency_Without_Hint']*100:.2f}%" if result['QA_Generation_Consistency_Without_Hint'] is not None else "  问答题一致性(无提示): N/A")
        print(f"  问答题一致性(有提示): {result['QA_Generation_Consistency_With_Hint']*100:.2f}%" if result['QA_Generation_Consistency_With_Hint'] is not None else "  问答题一致性(有提示): N/A")
        print(f"  未知问题拒答率: {result['Unknowns_Refusal_Rate']*100:.2f}%" if result['Unknowns_Refusal_Rate'] is not None else "  未知问题拒答率: N/A")
        print(f"  Bio MCQ拒绝率: {result['Bio_MCQ_Rejection_Rate']*100:.2f}%" if result['Bio_MCQ_Rejection_Rate'] is not None else "  Bio MCQ拒绝率: N/A")
        print(f"  Bio MCQ拒绝率标准差: {result['Bio_MCQ_Rejection_Rate_Std']*100:.2f}%" if result['Bio_MCQ_Rejection_Rate_Std'] is not None else "  Bio MCQ拒绝率标准差: N/A")
        print(f"  E选项(Sorry)拒绝率: {result['E_Option_Sorry_Rejection_Rate']*100:.2f}%" if result['E_Option_Sorry_Rejection_Rate'] is not None else "  E选项(Sorry)拒绝率: N/A")
        print(f"  E选项(Sorry)拒绝率标准差: {result['E_Option_Sorry_Rejection_Std']*100:.2f}%" if result['E_Option_Sorry_Rejection_Std'] is not None else "  E选项(Sorry)拒绝率标准差: N/A")
        print(f"  E选项(Space)拒绝率: {result['E_Option_Space_Rejection_Rate']*100:.2f}%" if result['E_Option_Space_Rejection_Rate'] is not None else "  E选项(Space)拒绝率: N/A")
        print(f"  E选项(Space)拒绝率标准差: {result['E_Option_Space_Rejection_Std']*100:.2f}%" if result['E_Option_Space_Rejection_Std'] is not None else "  E选项(Space)拒绝率标准差: N/A")
        print(f"  Bio MCQ词汇熵: {result['Bio_MCQ_Full_Vocab_Entropy']:.6f}" if result['Bio_MCQ_Full_Vocab_Entropy'] is not None else "  Bio MCQ词汇熵: N/A")
        print(f"  MMLU分数: {result['MMLU_Score']*100:.2f}%" if result['MMLU_Score'] is not None else "  MMLU分数: N/A")
        print(f"  WMDP Bio分数: {result['WMDP_Bio_Score']*100:.2f}%" if result['WMDP_Bio_Score'] is not None else "  WMDP Bio分数: N/A")
        print(f"  指令遵循分数: {result['Instruction_Following_Score']*100:.2f}%" if result['Instruction_Following_Score'] is not None else "  指令遵循分数: N/A")
        print(f"  一致性分数: {result['Consistency_Score']*100:.2f}%" if result['Consistency_Score'] is not None else "  一致性分数: N/A")
        print(f"  Unlearn 2Q一致性: {result['Unlearn_2Q_Consistency']*100:.2f}%" if result['Unlearn_2Q_Consistency'] is not None else "  Unlearn 2Q一致性: N/A")
        print()


if __name__ == "__main__":
    main()
