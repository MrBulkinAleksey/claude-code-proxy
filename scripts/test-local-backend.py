#!/usr/bin/env python3
"""Exercise a running local provider, including a complete tool round trip."""
import json
import os
import urllib.request

base = os.environ.get('CCP_TEST_URL', 'http://127.0.0.1:18766')
model = 'local/' + os.environ.get('CCP_LOCAL_MODELS', 'qwen3.8-flash-next-gyro-gyro-m').split(',')[0]

def request(messages, **options):
    body = dict(model=model, max_tokens=1024, messages=messages, **options)
    req = urllib.request.Request(base + '/v1/messages', json.dumps(body).encode(),
                                 {'Content-Type': 'application/json', 'anthropic-version': '2023-06-01'})
    with urllib.request.urlopen(req, timeout=180) as response:
        raw = response.read().decode()
    return raw if options.get('stream') else json.loads(raw)

simple = [{'role': 'user', 'content': 'Reply with exactly LOCAL_OK. No explanation.'}]
result = request(simple)
assert any('LOCAL_OK' in b.get('text', '') for b in result['content']), result
print('Buffered text: OK', flush=True)
stream = request(simple, stream=True)
assert 'event: message_start' in stream and 'event: message_stop' in stream, stream
events = [json.loads(line[6:]) for line in stream.splitlines() if line.startswith('data: ')]
text = ''.join(event.get('delta', {}).get('text', '') for event in events)
assert 'LOCAL_OK' in text, stream
print('Streaming text: OK', flush=True)
tools = [{'name': 'get_test_value', 'description': 'Return the test value for a key.',
          'input_schema': {'type': 'object', 'properties': {'key': {'type': 'string'}}, 'required': ['key']}}]
messages = [{'role': 'user', 'content': 'Call get_test_value with key="probe". Then report the value it returns.'}]
tool_stream = request(messages, tools=tools, tool_choice={'type': 'tool', 'name': 'get_test_value'}, stream=True)
assert 'tool_use' in tool_stream and 'input_json_delta' in tool_stream and 'event: message_stop' in tool_stream, tool_stream
print('Streaming tool call: OK', flush=True)
result = request(messages, tools=tools, tool_choice={'type': 'tool', 'name': 'get_test_value'})
calls = [b for b in result['content'] if b['type'] == 'tool_use']
assert calls and result['stop_reason'] == 'tool_use', result
assert calls[0]['name'] == 'get_test_value' and calls[0]['input']['key'] == 'probe', result
messages += [{'role': 'assistant', 'content': result['content']},
             {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': call['id'], 'content': 'VALUE_7391'} for call in calls]}]
result = request(messages, tools=tools, tool_choice={'type': 'none'})
assert any('VALUE_7391' in b.get('text', '') for b in result['content']), result
print('Tool call and tool result: OK', flush=True)
