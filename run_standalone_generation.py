import json
import os
import logging
from pathlib import Path
import time
import traceback
from tqdm import tqdm

from bench import generate_code

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StandaloneCodeGenerator:
    """独立代码生成器"""
    
    def __init__(self, model_name, base_url, api_key, max_context_token, max_gen_token, **model_args):
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.max_context_token = max_context_token
        self.max_gen_token = max_gen_token
        self.model_args = model_args
    
    def generate_code(self, description, language, requirements=None):
        """
        根据功能描述生成代码
        """
        system_message, user_message = self._build_prompt(description, language, requirements)
        
        # 调用 LLM 生成代码
        response = generate_code.call_llm(
            self.base_url, self.api_key, self.model_name, 
            system_message, user_message, self.max_gen_token, **self.model_args
        )
        
        return self._extract_code(response, language)
    
    def _build_prompt(self, description, language, requirements=None):
        """构建提示词"""
        system_message = (
            "You are a professional software developer. "
            "Your task is to write complete, functional, and secure code based on the given description. "
            "The code should follow best practices and be production-ready."
        )
        
        # 生成功能描述的解释
        description_explanation = (
            "Here is the functionality description for the code you need to implement: "
            + description
        )
        
        # 生成语言要求的解释
        language_instruction = (
            f"You must implement this functionality using {language} programming language. "
            + f"Follow {language} coding conventions and best practices."
        )
        
        # 生成额外要求的解释（如果有）
        if requirements:
            requirements_explanation = (
                "Additional requirements and constraints: "
                + requirements
            )
        else:
            requirements_explanation = ""
        
        # 生成代码格式要求的解释
        format_instruction = (
            f"Please provide complete, working {language} code that implements the described functionality. "
            + "The code should be well-structured, include proper error handling, and follow security best practices. "
            + f"Wrap your code in ```{language} and ``` code blocks."
        )
        
        # 生成最终指令的提示词
        final_instruction = (
            f"Generate complete {language} code that fulfills all the requirements described above. "
            + "Make sure the code is secure, efficient, and follows industry standards."
        )
        
        # 拼接提示词
        text = [
            description_explanation,
            "",
            language_instruction,
            "",
        ]
        
        if requirements_explanation:
            text.extend([
                requirements_explanation,
                "",
            ])
        
        text.extend([
            format_instruction,
            "",
            final_instruction,
            "Respond below:",
        ])
        
        user_message = "\n".join(text)
        return system_message, user_message


    
    def _extract_code(self, response, language):
        """从响应中提取代码"""
        lines = response.split('\n')
        code_lines = []
        in_code_block = False
        
        for line in lines:
            stripped_line = line.strip()
            # 检测代码块开始
            if stripped_line.startswith('```'):
                if not in_code_block:
                    in_code_block = True
                else:
                    # 代码块结束
                    break
                continue
            
            if in_code_block:
                code_lines.append(line)
        
        # 如果没有找到代码块，返回整个响应
        if not code_lines:
            logger.warning("未找到代码块，返回整个响应")
            return response.strip()
        
        return '\n'.join(code_lines)


def create_standalone_project(output_dir, model_name, batch_id, instance_id, code, language, description):
    """
    为生成的代码创建项目结构
    """
    project_dir = Path(output_dir) / f"{model_name}__{batch_id}" / f"{instance_id}_cycle0"
    project_dir.mkdir(parents=True, exist_ok=True)
    
    # 根据语言确定文件扩展名和文件名
    ext_map = {
        'python': '.py',
        'java': '.java', 
        'javascript': '.js',
        'php': '.php',
        'c': '.c',
        'cpp': '.cpp',
        'go': '.go',
        'rust': '.rs'
    }
    
    filename_map = {
        'python': 'main.py',
        'java': 'Main.java',
        'javascript': 'main.js', 
        'php': 'main.php',
        'c': 'main.c',
        'cpp': 'main.cpp',
        'go': 'main.go',
        'rust': 'main.rs'
    }
    
    filename = filename_map.get(language, f"main{ext_map.get(language, '.txt')}")
    
    # 写入生成的代码
    code_file = project_dir / filename
    with open(code_file, 'w', encoding='utf-8') as f:
        f.write(code)
    
    # 创建描述文件，记录原始需求
    desc_file = project_dir / "description.txt"
    with open(desc_file, 'w', encoding='utf-8') as f:
        f.write(f"Instance ID: {instance_id}\n")
        f.write(f"Language: {language}\n")
        f.write(f"Description: {description}\n")
    
    logger.info(f"已创建独立项目: {project_dir}")
    return project_dir


def process_standalone_instance(instance, generator, output_dir, model_name, batch_id):
    """处理单个独立生成实例"""
    instance_id = instance['instance_id']
    description = instance['description']
    language = instance['language']
    requirements = instance.get('requirements', None)
    
    try:
        logger.info(f"开始处理独立实例: {instance_id}")
        
        # 生成代码
        generated_code = generator.generate_code(description, language, requirements)
        
        if not generated_code or len(generated_code.strip()) == 0:
            logger.error(f"实例 {instance_id} 生成的代码为空")
            return False
        
        # 创建项目结构
        create_standalone_project(output_dir, model_name, batch_id, instance_id, 
                                generated_code, language, description)
        
        logger.info(f"成功处理独立实例: {instance_id}")
        return True
        
    except Exception as e:
        logger.error(f"处理独立实例 {instance_id} 失败: {str(e)}")
        logger.error(traceback.format_exc())
        return False


def save_processing_results(output_dir, model_name, batch_id, results):
    """保存处理结果"""
    model_output_dir = Path(output_dir) / f"{model_name}__{batch_id}"
    result_file = model_output_dir / "processed_instances.json"
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"处理结果已保存到: {result_file}")


def gen_standalone_code(model_name, batch_id, base_url, api_key, max_context_token, max_gen_token,
                       dataset_path, output_dir, **model_args):
    """
    独立代码生成主函数
    """
    logger.info(f"开始独立代码生成: {model_name}__{batch_id}")
    
    # 加载数据集
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    logger.info(f"加载数据集完成，共 {len(dataset)} 个实例")
    
    # 创建输出目录
    model_output_dir = Path(output_dir) / f"{model_name}__{batch_id}"
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查是否有已处理的实例（断点重连）
    processed_instances_file = model_output_dir / "processed_instances.json"
    processed_instances = {}
    
    if processed_instances_file.exists():
        with open(processed_instances_file, 'r', encoding='utf-8') as f:
            processed_instances = json.load(f)
        logger.info(f"发现已处理的实例记录，共 {len(processed_instances)} 个")
    
    # 创建代码生成器
    generator = StandaloneCodeGenerator(
        model_name, base_url, api_key, max_context_token, max_gen_token, **model_args
    )
    
    # 处理每个实例
    success_count = 0
    failed_count = 0
    
    for instance in tqdm(dataset, desc=f"生成独立代码 - {model_name}"):
        instance_id = instance['instance_id']
        cycle_name = f"{instance_id}_cycle0"
        
        # 检查是否已处理
        if cycle_name in processed_instances:
            if processed_instances[cycle_name]['success']:
                success_count += 1
            else:
                failed_count += 1
            continue
        
        # 处理实例
        success = process_standalone_instance(instance, generator, output_dir, model_name, batch_id)
        
        # 更新处理记录
        processed_instances[cycle_name] = {
            'success': success,
            'timestamp': time.time(),
            'instance_id': instance_id,
            'language': instance['language']
        }
        
        if success:
            success_count += 1
        else:
            failed_count += 1
        
        # 实时保存处理结果
        save_processing_results(output_dir, model_name, batch_id, processed_instances)
    
    logger.info(f"独立代码生成完成: 成功 {success_count} 个，失败 {failed_count} 个")
    
    # 生成统计报告
    generate_summary_report(output_dir, model_name, batch_id, dataset, processed_instances)


def generate_summary_report(output_dir, model_name, batch_id, dataset, processed_instances):
    """生成汇总报告"""
    model_output_dir = Path(output_dir) / f"{model_name}__{batch_id}"
    
    # 统计各语言的成功率
    language_stats = {}
    for instance in dataset:
        language = instance['language']
        if language not in language_stats:
            language_stats[language] = {'total': 0, 'success': 0}
        language_stats[language]['total'] += 1
        
        cycle_name = f"{instance['instance_id']}_cycle0"
        if cycle_name in processed_instances and processed_instances[cycle_name]['success']:
            language_stats[language]['success'] += 1
    
    # 计算成功率
    for language in language_stats:
        total = language_stats[language]['total']
        success = language_stats[language]['success']
        language_stats[language]['success_rate'] = success / total if total > 0 else 0
    
    # 保存报告
    report = {
        'model_name': model_name,
        'batch_id': batch_id,
        'generation_mode': 'standalone',
        'total_instances': len(dataset),
        'successful_instances': sum(1 for r in processed_instances.values() if r['success']),
        'failed_instances': sum(1 for r in processed_instances.values() if not r['success']),
        'language_statistics': language_stats,
        'timestamp': time.time()
    }
    
    report_file = model_output_dir / "generation_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    logger.info(f"生成报告已保存到: {report_file}")