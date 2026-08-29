// server.js - OpenAI to NVIDIA NIM API Proxy
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' }));

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// 🔥 REASONING DISPLAY TOGGLE - Shows/hides reasoning in output
const SHOW_REASONING = false;

// 🔥 THINKING MODE TOGGLE - Enables thinking for models that support it
const ENABLE_THINKING_MODE = true;

// Model mapping
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'z-ai/glm-5.2',
  'gpt-5': 'z-ai/glm-4_7',
  'gpt-5-turbo': 'z-ai/glm-4.7',
  'gpt-4': 'qwen/qwen3.5-122b-a10b',
  'gpt-4.5': 'qwen/qwen3.5-397b-a17b',
  'gpt-4-turbo': 'deepseek-ai/deepseek-v3.2',
  'gpt444': 'deepseek-ai/deepseek-v4-pro',
  'gpt-5.4': 'deepseek-ai/deepseek-v4-pro',
  'gpt-4o-mini': 'deepseek-ai/deepseek-v4-pro-0813',
  'claude-3-opus': 'openai/gpt-oss-120b',
  'claude-3-sonnet': 'openai/gpt-oss-20b',
  'gemini-pro': 'moonshotai/kimi-k3'
};

// -----------------------------
// Helpers
// -----------------------------

function redactSecrets(value) {
  if (typeof value !== 'string') return value;

  return value
    .replace(/Bearer\s+[A-Za-z0-9._\-]+/gi, 'Bearer [REDACTED]')
    .replace(/nvapi-[A-Za-z0-9._\-]+/gi, 'nvapi-[REDACTED]');
}

function parseMaybeJson(text) {
  if (typeof text !== 'string') return text;

  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

function safeStringify(value) {
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function getErrorMessage(body, fallback = 'Internal server error') {
  if (!body) return fallback;

  if (typeof body === 'string') return body;

  return (
    body.error?.message ||
    body.message ||
    body.detail ||
    body.title ||
    fallback
  );
}

async function streamToString(stream) {
  return new Promise((resolve, reject) => {
    let data = '';

    stream.on('data', chunk => {
      data += chunk.toString();
    });

    stream.on('end', () => resolve(data));
    stream.on('error', reject);
  });
}

async function readResponseBody(response) {
  if (!response) return null;

  let body = response.data;

  // Axios returns IncomingMessage for responseType: 'stream',
  // including error bodies from NVIDIA.
  if (body && typeof body.pipe === 'function') {
    const rawText = await streamToString(body);
    return parseMaybeJson(rawText);
  }

  return body;
}

async function logNvidiaError(response, label = 'NVIDIA error') {
  const body = await readResponseBody(response);
  const message = getErrorMessage(body, `${label}: ${response?.status || 500}`);

  console.error(`${label} status:`, response?.status || 500);
  console.error(`${label} message:`, message);

  if (response?.headers?.['nvcf-reqid']) {
    console.error(`${label} nvcf-reqid:`, response.headers['nvcf-reqid']);
  }

  if (body) {
    console.error(
      `${label} body:`,
      redactSecrets(typeof body === 'string' ? body : safeStringify(body))
    );
  }

  return { body, message };
}

function buildOpenAIError(status, message, body, headers) {
  return {
    error: {
      message: message || 'Internal server error',
      type: body?.error?.type || 'invalid_request_error',
      code: body?.error?.code || status || 500,
      nvcf_reqid: headers?.['nvcf-reqid'],
      details: body
    }
  };
}

function addThinkingParams(nimRequest, stream) {
  if (!ENABLE_THINKING_MODE) return;

  // Important:
  // For direct axios HTTP requests, do NOT wrap this in "extra_body".
  // "extra_body" is for SDK-style clients, not raw NVIDIA JSON requests.
  nimRequest.chat_template_kwargs = {
    enable_thinking: true
  };

  // Reasoning hiding for non-streaming only.
  // Some providers reject this for streaming.
  if (!stream && !SHOW_REASONING) {
    nimRequest.include_reasoning = false;
  }
}

function sanitizeOpenAIParams(reqBody) {
  const {
    model,
    messages,
    temperature,
    max_tokens,
    max_completion_tokens,
    stream,
    top_p,
    frequency_penalty,
    presence_penalty,
    stop
  } = reqBody;

  return {
    model,
    messages,
    temperature,
    max_tokens: max_tokens ?? max_completion_tokens,
    stream,
    top_p,
    frequency_penalty,
    presence_penalty,
    stop
  };
}

// -----------------------------
// Health check endpoint
// -----------------------------

app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy',
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE
  });
});

// -----------------------------
// List models endpoint
// -----------------------------

app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Math.floor(Date.now() / 1000),
    owned_by: 'nvidia-nim-proxy'
  }));

  res.json({
    object: 'list',
    data: models
  });
});

// -----------------------------
// Chat completions endpoint
// -----------------------------

app.post('/v1/chat/completions', async (req, res) => {
  try {
    if (!NIM_API_KEY) {
      return res.status(500).json({
        error: {
          message: 'Missing NIM_API_KEY environment variable',
          type: 'server_configuration_error',
          code: 500
        }
      });
    }

    const params = sanitizeOpenAIParams(req.body);

    const {
      model,
      messages,
      temperature,
      max_tokens,
      stream,
      top_p,
      frequency_penalty,
      presence_penalty,
      stop
    } = params;

    if (!model) {
      return res.status(400).json({
        error: {
          message: 'Missing required field: model',
          type: 'invalid_request_error',
          code: 400
        }
      });
    }

    if (!Array.isArray(messages)) {
      return res.status(400).json({
        error: {
          message: 'Missing or invalid required field: messages',
          type: 'invalid_request_error',
          code: 400
        }
      });
    }

    // Smart model selection with fallback
    let nimModel = MODEL_MAPPING[model];

    if (!nimModel) {
      try {
        const testResponse = await axios.post(
          `${NIM_API_BASE}/chat/completions`,
          {
            model,
            messages: [{ role: 'user', content: 'test' }],
            max_tokens: 1,
            stream: false
          },
          {
            headers: {
              Authorization: `Bearer ${NIM_API_KEY}`,
              'Content-Type': 'application/json'
            },
            validateStatus: status => status < 500
          }
        );

        if (testResponse.status >= 200 && testResponse.status < 300) {
          nimModel = model;
        } else {
          await logNvidiaError(testResponse, 'Model test error');
        }
      } catch (e) {
        if (e.response) {
          await logNvidiaError(e.response, 'Model test exception');
        } else {
          console.error('Model test exception:', redactSecrets(e.message));
        }
      }

      if (!nimModel) {
        const modelLower = model.toLowerCase();

        if (
          modelLower.includes('gpt-4') ||
          modelLower.includes('claude-opus') ||
          modelLower.includes('405b')
        ) {
          nimModel = 'meta/llama-3.1-405b-instruct';
        } else if (
          modelLower.includes('claude') ||
          modelLower.includes('gemini') ||
          modelLower.includes('70b')
        ) {
          nimModel = 'meta/llama-3.1-70b-instruct';
        } else {
          nimModel = 'meta/llama-3.1-8b-instruct';
        }
      }
    }

    // Transform OpenAI request to NVIDIA NIM format
    const nimRequest = {
      model: nimModel,
      messages,
      temperature: temperature ?? 0.6,
      max_tokens: max_tokens ?? 9024,
      stream: stream ?? false
    };

    if (top_p !== undefined) nimRequest.top_p = top_p;
    if (frequency_penalty !== undefined) nimRequest.frequency_penalty = frequency_penalty;
    if (presence_penalty !== undefined) nimRequest.presence_penalty = presence_penalty;
    if (stop !== undefined) nimRequest.stop = stop;

    addThinkingParams(nimRequest, Boolean(stream));

    // Make request to NVIDIA NIM API.
    // validateStatus: true prevents Axios from throwing before we can read the actual error body.
    const response = await axios.post(
      `${NIM_API_BASE}/chat/completions`,
      nimRequest,
      {
        headers: {
          Authorization: `Bearer ${NIM_API_KEY}`,
          'Content-Type': 'application/json'
        },
        responseType: stream ? 'stream' : 'json',
        validateStatus: () => true
      }
    );

    // Handle NVIDIA errors before stream/non-stream logic
    if (response.status >= 400) {
      const { body, message } = await logNvidiaError(response, 'NVIDIA API error');

      return res
        .status(response.status)
        .json(buildOpenAIError(response.status, message, body, response.headers));
    }

    // -----------------------------
    // Streaming response
    // -----------------------------

    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      let buffer = '';
      let reasoningStarted = false;

      response.data.on('data', chunk => {
        buffer += chunk.toString();

        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          const trimmed = line.trim();

          if (!trimmed) continue;

          if (!trimmed.startsWith('data: ')) {
            res.write(`${line}\n`);
            continue;
          }

          const payload = trimmed.slice(6);

          if (payload === '[DONE]') {
            if (SHOW_REASONING && reasoningStarted) {
              const closingChunk = {
                choices: [
                  {
                    delta: {
                      content: '\n</think>\n\n'
                    },
                    index: 0,
                    finish_reason: null
                  }
                ]
              };

              res.write(`data: ${JSON.stringify(closingChunk)}\n\n`);
              reasoningStarted = false;
            }

            res.write('data: [DONE]\n\n');
            continue;
          }

          try {
            const data = JSON.parse(payload);
            const delta = data.choices?.[0]?.delta;

            if (delta) {
              const reasoning = delta.reasoning_content;
              const content = delta.content;

              if (SHOW_REASONING) {
                let combinedContent = '';

                if (reasoning && !reasoningStarted) {
                  combinedContent += `<think>\n${reasoning}`;
                  reasoningStarted = true;
                } else if (reasoning) {
                  combinedContent += reasoning;
                }

                if (content && reasoningStarted) {
                  combinedContent += `\n</think>\n\n${content}`;
                  reasoningStarted = false;
                } else if (content) {
                  combinedContent += content;
                }

                delta.content = combinedContent;
                delete delta.reasoning_content;
              } else {
                // Hide reasoning from OpenAI-compatible clients.
                delta.content = content || '';
                delete delta.reasoning_content;
              }
            }

            res.write(`data: ${JSON.stringify(data)}\n\n`);
          } catch (e) {
            console.error('Failed to parse stream chunk:', redactSecrets(e.message));
            console.error('Raw stream line:', redactSecrets(line));
            res.write(`${line}\n\n`);
          }
        }
      });

      response.data.on('end', () => {
        res.end();
      });

      response.data.on('error', err => {
        console.error('NVIDIA stream error:', redactSecrets(err.message || String(err)));

        const errorPayload = {
          error: {
            message: err.message || 'NVIDIA stream error',
            type: 'stream_error',
            code: 500
          }
        };

        res.write(`data: ${JSON.stringify(errorPayload)}\n\n`);
        res.end();
      });

      return;
    }

    // -----------------------------
    // Non-streaming response
    // -----------------------------

    const choices = Array.isArray(response.data?.choices) ? response.data.choices : [];

    const openaiResponse = {
      id: response.data?.id || `chatcmpl-${Date.now()}`,
      object: 'chat.completion',
      created: response.data?.created || Math.floor(Date.now() / 1000),
      model,
      choices: choices.map((choice, index) => {
        const message = choice.message || {};
        let fullContent = message.content || '';

        if (SHOW_REASONING && message.reasoning_content) {
          fullContent = `<think>\n${message.reasoning_content}\n</think>\n\n${fullContent}`;
        }

        return {
          index: choice.index ?? index,
          message: {
            role: message.role || 'assistant',
            content: fullContent
          },
          finish_reason: choice.finish_reason || null
        };
      }),
      usage: response.data?.usage || {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0
      }
    };

    res.json(openaiResponse);
  } catch (error) {
    let body = null;
    let status = 500;
    let headers = {};

    if (error.response) {
      status = error.response.status || 500;
      headers = error.response.headers || {};
      body = await readResponseBody(error.response);
    }

    const message = getErrorMessage(body, error.message || 'Internal server error');

    console.error('Proxy exception status:', status);
    console.error('Proxy exception message:', redactSecrets(message));

    if (headers?.['nvcf-reqid']) {
      console.error('Proxy exception nvcf-reqid:', headers['nvcf-reqid']);
    }

    if (body) {
      console.error(
        'Proxy exception body:',
        redactSecrets(typeof body === 'string' ? body : safeStringify(body))
      );
    }

    res
      .status(status)
      .json(buildOpenAIError(status, message, body, headers));
  }
});

// -----------------------------
// Catch-all for unsupported endpoints
// -----------------------------

app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

app.listen(PORT, () => {
  console.log(`OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
  console.log(`Reasoning display: ${SHOW_REASONING ? 'ENABLED' : 'DISABLED'}`);
  console.log(`Thinking mode: ${ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED'}`);
});
