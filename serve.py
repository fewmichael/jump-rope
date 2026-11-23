# 本地预览：python serve.py 然后浏览器打开 http://localhost:8765
import http.server, functools
H = http.server.SimpleHTTPRequestHandler
H.extensions_map.update({'.mjs': 'text/javascript', '.js': 'text/javascript', '.wasm': 'application/wasm', '.task': 'application/octet-stream'})
http.server.ThreadingHTTPServer(('0.0.0.0', 8765), functools.partial(H, directory=__import__('os').path.dirname(__file__) or '.')).serve_forever()
