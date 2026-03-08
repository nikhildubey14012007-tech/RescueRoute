const http = require('http');

const server = http.createServer((req, res) => {
    res.setHeader('Content-Type', 'application/json');
    
    if (req.url === '/' && req.method === 'GET') {
        res.writeHead(200);
        res.end(JSON.stringify({message: 'Server is running', port: 5000}));
    } else {
        res.writeHead(404);
        res.end(JSON.stringify({error: '404 Not Found', path: req.url}));
    }
});

server.listen(5000, () => {
  console.log("Server is running at http://localhost:5000");
});