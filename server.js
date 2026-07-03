// server.js - Echo / Debug Proxy
const express = require('express');
const cors = require('cors');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '100mb' })); 
app.use(express.urlencoded({ limit: '100mb', extended: true }));

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    service: 'Echo Debug Server',
    mode: 'Returning requests directly without calling NVIDIA'
  });
});

// List models endpoint (OpenAI compatible)
app.get('/v1/models', (req, res) => {
  res.json({
    object: 'list',
    data: [
      { id: 'debug-model', object: 'model', created: Date.now(), owned_by: 'echo-proxy' }
    ]
  });
});

// Chat completions endpoint - ECHO MODE
app.post('/v1/chat/completions', (req, res) => {
  console.log('--- Request Received ---');
  console.log('Headers:', JSON.stringify(req.headers, null, 2));
  console.log('Body:', JSON.stringify(req.body, null, 2));
  console.log('------------------------');

  // Simply return the request body back to the client
  res.status(200).json({
    status: 'success',
    message: 'Request successfully echoed back. No external API was called.',
    received_request: req.body
  });
});

// Catch-all for unsupported endpoints
app.all('*', (req, res) => {
  console.log(`[404] Endpoint not found: ${req.method} ${req.path}`);
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

app.listen(PORT, () => {
  console.log(`Echo Debug Server running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
  console.log(`Send a POST to http://localhost:${PORT}/v1/chat/completions to see it echo back`);
});
