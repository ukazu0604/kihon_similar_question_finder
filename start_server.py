import http.server
import socketserver
import os

PORT = 8000

class MyHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # リクエストされたパスをログに出力
        print(f"Serving: {self.path}")
        return http.server.SimpleHTTPRequestHandler.do_GET(self)

# 現在のスクリプトのディレクトリをWebサーバーのルートにする
web_dir = os.path.join(os.path.dirname(__file__))
os.chdir(web_dir)

Handler = MyHandler
httpd = socketserver.TCPServer(("", PORT), Handler)

print(f"Serving at http://localhost:{PORT}")
print("Press Ctrl+C to stop the server.")

try:
    httpd.serve_forever()
except KeyboardInterrupt:
    print("\nServer stopped.")
    httpd.shutdown()
    socketserver.TCPServer.server_close(httpd)
