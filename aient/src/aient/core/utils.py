import re
import io
import ast
import json
import httpx
import base64
import asyncio
from time import time
from PIL import Image
from fastapi import HTTPException
from urllib.parse import urlparse
from collections import defaultdict

from .log_config import logger

def get_model_dict(provider):
    model_dict = {}
    if "model" not in provider:
        logger.error(f"Error: model is not set in provider: {provider}")
        return model_dict
    for model in provider['model']:
        if isinstance(model, str):
            model_dict[model] = model
        if isinstance(model, dict):
            model_dict.update({str(new): old for old, new in model.items()})
    return model_dict

class BaseAPI:
    def __init__(
        self,
        api_url: str = "https://api.openai.com/v1/chat/completions",
    ):
        if api_url == "":
            api_url = "https://api.openai.com/v1/chat/completions"
        self.source_api_url: str = api_url
        from urllib.parse import urlparse, urlunparse
        parsed_url = urlparse(self.source_api_url)
        # print("parsed_url", parsed_url)
        if parsed_url.scheme == "":
            raise Exception("Error: API_URL is not set")
        if parsed_url.path != '/':
            before_v1 = parsed_url.path.split("chat/completions")[0]
            if not before_v1.endswith("/"):
                before_v1 = before_v1 + "/"
        else:
            before_v1 = ""
        self.base_url: str = urlunparse(parsed_url[:2] + ("",) + ("",) * 3)
        self.v1_url: str = urlunparse(parsed_url[:2]+ (before_v1,) + ("",) * 3)
        self.v1_models: str = urlunparse(parsed_url[:2] + (before_v1 + "models",) + ("",) * 3)
        self.chat_url: str = urlunparse(parsed_url[:2] + (before_v1 + "chat/completions",) + ("",) * 3)
        self.image_url: str = urlunparse(parsed_url[:2] + (before_v1 + "images/generations",) + ("",) * 3)
        self.audio_transcriptions: str = urlunparse(parsed_url[:2] + (before_v1 + "audio/transcriptions",) + ("",) * 3)
        self.moderations: str = urlunparse(parsed_url[:2] + (before_v1 + "moderations",) + ("",) * 3)
        self.embeddings: str = urlunparse(parsed_url[:2] + (before_v1 + "embeddings",) + ("",) * 3)
        self.audio_speech: str = urlunparse(parsed_url[:2] + (before_v1 + "audio/speech",) + ("",) * 3)

        if parsed_url.hostname == "generativelanguage.googleapis.com":
            self.base_url = api_url
            self.v1_url = api_url
            self.chat_url = api_url
            self.embeddings = api_url

from urllib.parse import urlparse

from urllib.parse import urlparse

def get_engine(provider, endpoint=None, original_model=""):
    base_url = provider['base_url']
    if isinstance(base_url, bytes):
        base_url = base_url.decode("utf-8")  # 🔧 decode bytes to str

    parsed_url = urlparse(base_url)

    engine = None
    stream = None

    if path.endswith("/v1beta") or \
       path.endswith("/v1") or \
       (parsed_url.netloc == 'generativelanguage.googleapis.com' and "openai/chat/completions" not in path):
        engine = "gemini"
    elif parsed_url.netloc.rstrip('/').endswith('aiplatform.googleapis.com') or \
         (parsed_url.netloc.rstrip('/').endswith('gateway.ai.cloudflare.com') and "google-vertex-ai" in path) or \
         "aiplatform.googleapis.com" in path:
        engine = "vertex"
    elif parsed_url.netloc.rstrip('/').endswith('azure.com'):
        engine = "azure"
    elif parsed_url.netloc.rstrip('/').endswith('azuredatabricks.net'):
        engine = "azure-databricks"
    elif parsed_url.netloc == 'api.cloudflare.com':
        engine = "cloudflare"
    elif parsed_url.netloc == 'api.anthropic.com' or path.endswith("v1/messages"):
        engine = "claude"
    elif 'amazonaws.com' in parsed_url.netloc:
        engine = "aws"
    elif parsed_url.netloc == 'api.cohere.com':
        engine = "cohere"
        stream = True
    else:
        engine = "gpt"

    return engine, stream

    original_model = original_model.lower()
    if original_model \
    and "claude" not in original_model \
    and "gpt" not in original_model \
    and "deepseek" not in original_model \
    and "o1" not in original_model \
    and "o3" not in original_model \
    and "o4" not in original_model \
    and "gemini" not in original_model \
    and "gemma" not in original_model \
    and "learnlm" not in original_model \
    and "grok" not in original_model \
    and parsed_url.netloc != 'api.cloudflare.com' \
    and parsed_url.netloc != 'api.cohere.com':
        engine = "openrouter"

    if "claude" in original_model and engine == "vertex":
        engine = "vertex-claude"

    if "gemini" in original_model and engine == "vertex":
        engine = "vertex-gemini"

    if provider.get("engine"):
        engine = provider["engine"]

    if engine != "gemini" and (endpoint == "/v1/images/generations" or "stable-diffusion" in original_model):
        engine = "dalle"
        stream = False

    if endpoint == "/v1/audio/transcriptions":
        engine = "whisper"
        stream = False

    if endpoint == "/v1/moderations":
        engine = "moderation"
        stream = False

    if endpoint == "/v1/embeddings":
        engine = "embedding"

    if endpoint == "/v1/audio/speech":
        engine = "tts"
        stream = False

    if "stream" in safe_get(provider, "preferences", "post_body_parameter_overrides", default={}):
        stream = safe_get(provider, "preferences", "post_body_parameter_overrides", "stream")

    return engine, stream

from httpx_socks import AsyncProxyTransport
def get_proxy(proxy, client_config = {}):
    if proxy:
        # 解析代理URL
        parsed = urlparse(proxy)
        scheme = parsed.scheme.rstrip('h')

        if scheme == 'socks5':
            proxy = proxy.replace('socks5h://', 'socks5://')
            transport = AsyncProxyTransport.from_url(proxy)
            client_config["transport"] = transport
            # print("proxy", proxy)
        else:
            client_config["proxies"] = {
                "http://": proxy,
                "https://": proxy
            }
    return client_config

def update_initial_model(provider):
    try:
        engine, stream_mode = get_engine(provider, endpoint=None, original_model="")
        # print("engine", engine, provider)
        api_url = provider['base_url']
        api = provider['api']
        proxy = safe_get(provider, "preferences", "proxy", default=None)
        client_config = get_proxy(proxy)
        if engine == "gemini":
            before_v1 = api_url.split("/v1beta")[0]
            url = before_v1 + "/v1beta/models"
            params = {"key": api}
            with httpx.Client(**client_config) as client:
                response = client.get(url, params=params)

            original_models = response.json()
            if original_models.get("error"):
                raise Exception({"error": original_models.get("error"), "endpoint": url, "api": api})

            models = {"data": []}
            for model in original_models["models"]:
                models["data"].append({
                    "id": model["name"].split("models/")[-1],
                })
        else:
            endpoint = BaseAPI(api_url=api_url)
            endpoint_models_url = endpoint.v1_models
            if isinstance(api, list):
                api = api[0]
            headers = {"Authorization": f"Bearer {api}"}
            response = httpx.get(
                endpoint_models_url,
                headers=headers,
                **client_config
            )
            models = response.json()
            if models.get("error"):
                logger.error({"error": models.get("error"), "endpoint": endpoint_models_url, "api": api})
                return []

        # print(models)
        models_list = models["data"]
        models_id = [model["id"] for model in models_list]
        set_models = set()
        for model_item in models_id:
            set_models.add(model_item)
        models_id = list(set_models)
        # print(models_id)
        return models_id
    except Exception as e:
        # print("error:", e)
        import traceback
        traceback.print_exc()
        return []

def safe_get(data, *keys, default=None):
    for key in keys:
        try:
            data = data[key] if isinstance(data, (dict, list)) else data.get(key)
        except (KeyError, IndexError, AttributeError, TypeError):
            return default
    if not data:
        return default
    return data

def parse_rate_limit(limit_string):
    # 定义时间单位到秒的映射
    time_units = {
        's': 1, 'sec': 1, 'second': 1,
        'm': 60, 'min': 60, 'minute': 60,
        'h': 3600, 'hr': 3600, 'hour': 3600,
        'd': 86400, 'day': 86400,
        'mo': 2592000, 'month': 2592000,
        'y': 31536000, 'year': 31536000
    }

    # 处理多个限制条件
    limits = []
    for limit in limit_string.split(','):
        limit = limit.strip()
        # 使用正则表达式匹配数字和单位
        match = re.match(r'^(\d+)/(\w+)$', limit)
        if not match:
            raise ValueError(f"Invalid rate limit format: {limit}")

        count, unit = match.groups()
        count = int(count)

        # 转换单位到秒
        if unit not in time_units:
            raise ValueError(f"Unknown time unit: {unit}")

        seconds = time_units[unit]
        limits.append((count, seconds))

    return limits

class ThreadSafeCircularList:
    def __init__(self, items = [], rate_limit={"default": "999999/min"}, schedule_algorithm="round_robin"):
        if schedule_algorithm == "random":
            import random
            self.items = random.sample(items, len(items))
            self.schedule_algorithm = "random"
        elif schedule_algorithm == "round_robin":
            self.items = items
            self.schedule_algorithm = "round_robin"
        elif schedule_algorithm == "fixed_priority":
            self.items = items
            self.schedule_algorithm = "fixed_priority"
        else:
            self.items = items
            logger.warning(f"Unknown schedule algorithm: {schedule_algorithm}, use (round_robin, random, fixed_priority) instead")
            self.schedule_algorithm = "round_robin"
        self.index = 0
        self.lock = asyncio.Lock()
        # 修改为二级字典，第一级是item，第二级是model
        self.requests = defaultdict(lambda: defaultdict(list))
        self.cooling_until = defaultdict(float)
        self.rate_limits = {}
        if isinstance(rate_limit, dict):
            for rate_limit_model, rate_limit_value in rate_limit.items():
                self.rate_limits[rate_limit_model] = parse_rate_limit(rate_limit_value)
        elif isinstance(rate_limit, str):
            self.rate_limits["default"] = parse_rate_limit(rate_limit)
        else:
            logger.error(f"Error ThreadSafeCircularList: Unknown rate_limit type: {type(rate_limit)}, rate_limit: {rate_limit}")

    async def set_cooling(self, item: str, cooling_time: int = 60):
        """设置某个 item 进入冷却状态

        Args:
            item: 需要冷却的 item
            cooling_time: 冷却时间(秒)，默认60秒
        """
        if item == None:
            return
        now = time()
        async with self.lock:
            self.cooling_until[item] = now + cooling_time
            # 清空该 item 的请求记录
            # self.requests[item] = []
            logger.warning(f"API key {item} 已进入冷却状态，冷却时间 {cooling_time} 秒")

    async def is_rate_limited(self, item, model: str = None, is_check: bool = False) -> bool:
        now = time()
        # 检查是否在冷却中
        if now < self.cooling_until[item]:
            return True

        # 获取适用的速率限制

        if model:
            model_key = model
        else:
            model_key = "default"

        rate_limit = None
        # 先尝试精确匹配
        if model and model in self.rate_limits:
            rate_limit = self.rate_limits[model]
        else:
            # 如果没有精确匹配，尝试模糊匹配
            for limit_model in self.rate_limits:
                if limit_model != "default" and model and limit_model in model:
                    rate_limit = self.rate_limits[limit_model]
                    break

        # 如果都没匹配到，使用默认值
        if rate_limit is None:
            rate_limit = self.rate_limits.get("default", [(999999, 60)])  # 默认限制

        # 检查所有速率限制条件
        for limit_count, limit_period in rate_limit:
            # 使用特定模型的请求记录进行计算
            recent_requests = sum(1 for req in self.requests[item][model_key] if req > now - limit_period)
            if recent_requests >= limit_count:
                if not is_check:
                    logger.warning(f"API key {item}: model: {model_key} has been rate limited ({limit_count}/{limit_period} seconds)")
                return True

        # 清理太旧的请求记录
        max_period = max(period for _, period in rate_limit)
        self.requests[item][model_key] = [req for req in self.requests[item][model_key] if req > now - max_period]

        # 记录新的请求
        if not is_check:
            self.requests[item][model_key].append(now)

        return False

    async def next(self, model: str = None):
        async with self.lock:
            if self.schedule_algorithm == "fixed_priority":
                self.index = 0
            start_index = self.index
            while True:
                item = self.items[self.index]
                self.index = (self.index + 1) % len(self.items)

                if not await self.is_rate_limited(item, model):
                    return item

                # 如果已经检查了所有的 API key 都被限制
                if self.index == start_index:
                    logger.warning(f"All API keys are rate limited!")
                    raise HTTPException(status_code=429, detail="Too many requests")

    async def is_all_rate_limited(self, model: str = None) -> bool:
        """检查是否所有的items都被速率限制

        与next方法不同，此方法不会改变任何内部状态（如self.index），
        仅返回一个布尔值表示是否所有的key都被限制。

        Args:
            model: 要检查的模型名称，默认为None

        Returns:
            bool: 如果所有items都被速率限制返回True，否则返回False
        """
        if len(self.items) == 0:
            return False

        async with self.lock:
            for item in self.items:
                if not await self.is_rate_limited(item, model, is_check=True):
                    return False

            # 如果遍历完所有items都被限制，返回True
            # logger.debug(f"Check result: all items are rate limited!")
            return True

    async def after_next_current(self):
        # 返回当前取出的 API，因为已经调用了 next，所以当前API应该是上一个
        if len(self.items) == 0:
            return None
        async with self.lock:
            item = self.items[(self.index - 1) % len(self.items)]
            return item

    def get_items_count(self) -> int:
        """返回列表中的项目数量

        Returns:
            int: items列表的长度
        """
        return len(self.items)

def circular_list_encoder(obj):
    if isinstance(obj, ThreadSafeCircularList):
        return obj.to_dict()
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

provider_api_circular_list = defaultdict(ThreadSafeCircularList)

# 【GCP-Vertex AI 目前有這些區域可用】 https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/use-claude?hl=zh_cn
# https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations?hl=zh-cn#available-regions

# c3.5s
# us-east5
# europe-west1

# c3s
# us-east5
# us-central1
# asia-southeast1

# c3o
# us-east5

# c3h
# us-east5
# us-central1
# europe-west1
# europe-west4


c35s = ThreadSafeCircularList(["us-east5", "europe-west1"])
c3s = ThreadSafeCircularList(["us-east5", "us-central1", "asia-southeast1"])
c3o = ThreadSafeCircularList(["us-east5"])
c4 = ThreadSafeCircularList(["us-east5", "us-central1", "europe-west4", "asia-southeast1"])
c3h = ThreadSafeCircularList(["us-east5", "us-central1", "europe-west1", "europe-west4"])
gemini1 = ThreadSafeCircularList(["us-central1", "us-east4", "us-west1", "us-west4", "europe-west1", "europe-west2"])
gemini2 = ThreadSafeCircularList(["us-central1"])
gemini2_5_pro_exp = ThreadSafeCircularList(["global"])



# end_of_line = "\n\r\n"
# end_of_line = "\r\n"
# end_of_line = "\n\r"
end_of_line = "\n\n"
# end_of_line = "\r"
# end_of_line = "\n"

import random
import string
async def generate_sse_response(timestamp, model, content=None, tools_id=None, function_call_name=None, function_call_content=None, role=None, total_tokens=0, prompt_tokens=0, completion_tokens=0, reasoning_content=None, stop=None):
    random.seed(timestamp)
    random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=29))

    delta_content = {"role": "assistant", "content": content} if content else {}
    if reasoning_content:
        delta_content = {"role": "assistant", "content": "", "reasoning_content": reasoning_content}

    sample_data = {
        "id": f"chatcmpl-{random_str}",
        "object": "chat.completion.chunk",
        "created": timestamp,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta_content,
                "logprobs": None,
                "finish_reason": None if content or reasoning_content else "stop"
            }
        ],
        "usage": None,
        "system_fingerprint": "fp_d576307f90",
    }
    if function_call_content:
        sample_data["choices"][0]["delta"] = {"tool_calls":[{"index":0,"function":{"arguments": function_call_content}}]}
    if tools_id and function_call_name:
        sample_data["choices"][0]["delta"] = {"tool_calls":[{"index":0,"id": tools_id,"type":"function","function":{"name": function_call_name, "arguments":""}}]}
        # sample_data["choices"][0]["delta"] = {"tool_calls":[{"index":0,"function":{"id": tools_id, "name": function_call_name}}]}
    if role:
        sample_data["choices"][0]["delta"] = {"role": role, "content": ""}
    if total_tokens:
        sample_data["usage"] = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": total_tokens}
        sample_data["choices"] = []
    if stop:
        sample_data["choices"][0]["delta"] = {}
        sample_data["choices"][0]["finish_reason"] = stop

    json_data = json.dumps(sample_data, ensure_ascii=False)
    # print("json_data", json.dumps(sample_data, indent=4, ensure_ascii=False))

    # 构建SSE响应
    sse_response = f"data: {json_data}" + end_of_line

    return sse_response

async def generate_no_stream_response(timestamp, model, content=None, tools_id=None, function_call_name=None, function_call_content=None, role=None, total_tokens=0, prompt_tokens=0, completion_tokens=0, reasoning_content=None, image_base64=None):
    random.seed(timestamp)
    random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=29))
    message = {
        "role": role,
        "content": content,
        "refusal": None
    }
    if reasoning_content:
        message["reasoning_content"] = reasoning_content

    sample_data = {
        "id": f"chatcmpl-{random_str}",
        "object": "chat.completion",
        "created": timestamp,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "logprobs": None,
                "finish_reason": "stop"
            }
        ],
        "usage": None,
        "system_fingerprint": "fp_a7d06e42a7"
    }

    if function_call_name:
        if not tools_id:
            tools_id = f"call_{random_str}"
        sample_data = {
            "id": f"chatcmpl-{random_str}",
            "object": "chat.completion",
            "created": timestamp,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tools_id,
                            "type": "function",
                            "function": {
                                "name": function_call_name,
                                "arguments": json.dumps(function_call_content, ensure_ascii=False)
                            }
                        }
                    ],
                    "refusal": None
                    },
                    "logprobs": None,
                    "finish_reason": "tool_calls"
                }
            ],
            "usage": None,
            "service_tier": "default",
            "system_fingerprint": "fp_4691090a87"
        }

    if image_base64:
        sample_data = {
            "created": timestamp,
            "data": [{
                "b64_json": image_base64
            }],
            # "usage": {
            #     "total_tokens": 100,
            #     "input_tokens": 50,
            #     "output_tokens": 50,
            # }
        }

    if total_tokens:
        sample_data["usage"] = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": total_tokens}

    json_data = json.dumps(sample_data, ensure_ascii=False)
    # print("json_data", json.dumps(sample_data, indent=4, ensure_ascii=False))

    return json_data

def get_image_format(file_content: bytes):
    try:
        img = Image.open(io.BytesIO(file_content))
        return img.format.lower()
    except:
        return None

def encode_image(file_content: bytes):
    img_format = get_image_format(file_content)
    if not img_format:
        raise ValueError("无法识别的图片格式")
    base64_encoded = base64.b64encode(file_content).decode('utf-8')

    if img_format == 'png':
        return f"data:image/png;base64,{base64_encoded}"
    elif img_format in ['jpg', 'jpeg']:
        return f"data:image/jpeg;base64,{base64_encoded}"
    else:
        raise ValueError(f"不支持的图片格式: {img_format}")

async def get_image_from_url(url):
    transport = httpx.AsyncHTTPTransport(
        http2=True,
        verify=False,
        retries=1
    )
    async with httpx.AsyncClient(transport=transport) as client:
        try:
            response = await client.get(
                url,
                timeout=30.0
            )
            response.raise_for_status()
            return response.content

        except httpx.RequestError as e:
            logger.error(f"请求 URL 时出错 {e.request.url!r}: {e}")
            raise HTTPException(status_code=400, detail=f"无法从 URL 获取内容: {url}")
        except httpx.HTTPStatusError as e:
            logger.error(f"获取 URL 时发生 HTTP 错误 {e.request.url!r}: {e.response.status_code}")
            raise HTTPException(status_code=e.response.status_code, detail=f"获取 URL 时出错: {url}")

async def get_encode_image(image_url):
    file_content = await get_image_from_url(image_url)
    base64_image = encode_image(file_content)
    return base64_image

# from PIL import Image
# import io
# def validate_image(image_data, image_type):
#     try:
#         decoded_image = base64.b64decode(image_data)
#         image = Image.open(io.BytesIO(decoded_image))

#         # 检查图片格式是否与声明的类型匹配
#         # print("image.format", image.format)
#         if image_type == "image/png" and image.format != "PNG":
#             raise ValueError("Image is not a valid PNG")
#         elif image_type == "image/jpeg" and image.format not in ["JPEG", "JPG"]:
#             raise ValueError("Image is not a valid JPEG")

#         # 如果没有异常,则图片有效
#         return True
#     except Exception as e:
#         print(f"Image validation failed: {str(e)}")
#         return False

async def get_image_message(base64_image, engine = None):
    if base64_image.startswith("http"):
        base64_image = await get_encode_image(base64_image)
    colon_index = base64_image.index(":")
    semicolon_index = base64_image.index(";")
    image_type = base64_image[colon_index + 1:semicolon_index]

    if image_type == "image/webp":
        # 将webp转换为png

        # 解码base64获取图片数据
        image_data = base64.b64decode(base64_image.split(",")[1])

        # 使用PIL打开webp图片
        image = Image.open(io.BytesIO(image_data))

        # 转换为PNG格式
        png_buffer = io.BytesIO()
        image.save(png_buffer, format="PNG")
        png_base64 = base64.b64encode(png_buffer.getvalue()).decode('utf-8')

        # 返回PNG格式的base64
        base64_image = f"data:image/png;base64,{png_base64}"
        image_type = "image/png"

    if "gpt" == engine or "openrouter" == engine or "azure" == engine or "azure-databricks" == engine:
        return {
            "type": "image_url",
            "image_url": {
                "url": base64_image,
            }
        }
    if "claude" == engine or "vertex-claude" == engine or "aws" == engine:
        # if not validate_image(base64_image.split(",")[1], image_type):
        #     raise ValueError(f"Invalid image format. Expected {image_type}")
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": image_type,
                "data": base64_image.split(",")[1],
            }
        }
    if "gemini" == engine or "vertex-gemini" == engine:
        return {
            "inlineData": {
                "mimeType": image_type,
                "data": base64_image.split(",")[1],
            }
        }
    raise ValueError("Unknown engine")

async def get_text_message(message, engine = None):
    if "gpt" == engine or "claude" == engine or "openrouter" == engine or \
    "vertex-claude" == engine or "azure" == engine or "aws" == engine or \
    "azure-databricks" == engine:
        return {"type": "text", "text": message}
    if "gemini" == engine or "vertex-gemini" == engine:
        return {"text": message}
    if engine == "cloudflare":
        return message
    if engine == "cohere":
        return message
    raise ValueError("Unknown engine")

def parse_json_safely(json_str):
    """
    尝试解析JSON字符串，先使用ast.literal_eval，失败则使用json.loads

    Args:
        json_str: 要解析的JSON字符串

    Returns:
        解析后的Python对象

    Raises:
        Exception: 当两种方法都失败时抛出异常
    """
    try:
        # 首先尝试使用ast.literal_eval解析
        return ast.literal_eval(json_str)
    except (SyntaxError, ValueError):
        try:
            # 如果失败，尝试使用json.loads解析
            return json.loads(json_str, strict=False)
        except json.JSONDecodeError as e:
            # 两种方法都失败，抛出异常
            raise Exception(f"无法解析JSON字符串: {e}, {json_str}")

if __name__ == "__main__":
    provider = {
        "base_url": "https://gateway.ai.cloudflare.com/v1/%7Baccount_id%7D/%7Bgateway_id%7D/google-vertex-ai",
    }
    print(get_engine(provider))
