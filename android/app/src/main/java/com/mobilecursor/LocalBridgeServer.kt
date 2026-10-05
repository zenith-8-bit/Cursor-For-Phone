package com.mobilecursor

import java.io.BufferedReader
import java.io.InputStreamReader
import java.io.OutputStream
import java.net.InetAddress
import java.net.ServerSocket
import java.net.Socket
import org.json.JSONObject

class LocalBridgeServer(
    private val service: CursorAccessibilityService,
    private val port: Int
) {
    @Volatile private var running = false
    private var serverSocket: ServerSocket? = null
    private var thread: Thread? = null

    fun start() {
        if (running) return
        running = true
        thread = Thread {
            try {
                serverSocket = ServerSocket(port, 50, InetAddress.getByName("127.0.0.1"))
                while (running) {
                    val socket = serverSocket?.accept() ?: break
                    Thread { handle(socket) }.start()
                }
            } catch (_: Exception) {
            }
        }.apply {
            isDaemon = true
            start()
        }
    }

    fun stop() {
        running = false
        try { serverSocket?.close() } catch (_: Exception) {}
        thread = null
    }

    private fun handle(socket: Socket) {
        socket.use { s ->
            try {
                val reader = BufferedReader(InputStreamReader(s.getInputStream()))
                val requestLine = reader.readLine() ?: return
                val headers = HashMap<String, String>()
                while (true) {
                    val line = reader.readLine() ?: break
                    if (line.isEmpty()) break
                    val idx = line.indexOf(':')
                    if (idx > 0) {
                        headers[line.substring(0, idx).trim().lowercase()] =
                            line.substring(idx + 1).trim()
                    }
                }

                val contentLength = headers["content-length"]?.toIntOrNull() ?: 0
                val bodyChars = CharArray(contentLength)
                var read = 0
                while (read < contentLength) {
                    val n = reader.read(bodyChars, read, contentLength - read)
                    if (n <= 0) break
                    read += n
                }
                val body = String(bodyChars, 0, read)

                val parts = requestLine.split(" ")
                val method = parts.getOrNull(0) ?: "GET"
                val path = parts.getOrNull(1) ?: "/"

                when {
                    method == "GET" && path == "/health" ->
                        respondJson(s, JSONObject().put("ok", true).put("service", "accessibility"))

                    method == "GET" && path == "/state" ->
                        respondJson(s, service.currentState())

                    method == "GET" && path == "/xml" ->
                        respondText(s, service.hierarchyXml(), "application/xml")

                    method == "POST" && path == "/action" ->
                        respondJson(s, service.executeAction(JSONObject(body)))

                    else ->
                        respondJson(
                            s,
                            JSONObject().put("ok", false).put("message", "not found")
                        )
                }
            } catch (e: Exception) {
                try {
                    respondJson(
                        s,
                        JSONObject().put("ok", false).put("message", e.message ?: "server error")
                    )
                } catch (_: Exception) {}
            }
        }
    }

    private fun respondJson(socket: Socket, json: JSONObject) {
        respondText(socket, json.toString(), "application/json")
    }

    private fun respondText(socket: Socket, body: String, contentType: String) {
        val out: OutputStream = socket.getOutputStream()
        val bytes = body.toByteArray(Charsets.UTF_8)
        val header = buildString {
            append("HTTP/1.1 200 OK\r\n")
            append("Content-Type: $contentType; charset=utf-8\r\n")
            append("Content-Length: ${bytes.size}\r\n")
            append("Connection: close\r\n\r\n")
        }.toByteArray(Charsets.UTF_8)
        out.write(header)
        out.write(bytes)
        out.flush()
    }
}
