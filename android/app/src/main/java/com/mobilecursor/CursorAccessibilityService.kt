package com.mobilecursor

import android.accessibilityservice.AccessibilityService
import android.accessibilityservice.GestureDescription
import android.graphics.Path
import android.graphics.Rect
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.accessibility.AccessibilityEvent
import android.view.accessibility.AccessibilityNodeInfo
import org.json.JSONArray
import org.json.JSONObject
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.max

class CursorAccessibilityService : AccessibilityService() {

    companion object {
        @Volatile var instance: CursorAccessibilityService? = null
    }

    private val revision = AtomicLong(0)
    private val nodeMap = ConcurrentHashMap<String, AccessibilityNodeInfo>()
    private val handler = Handler(Looper.getMainLooper())
    private lateinit var server: LocalBridgeServer

    @Volatile private var latestState: JSONObject = JSONObject()

    override fun onServiceConnected() {
        super.onServiceConnected()
        instance = this
        server = LocalBridgeServer(this, 8765)
        server.start()
        publishState()
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {
        publishState()
    }

    override fun onInterrupt() {}

    override fun onDestroy() {
        if (::server.isInitialized) server.stop()
        nodeMap.values.forEach { it.recycle() }
        nodeMap.clear()
        instance = null
        super.onDestroy()
    }

    fun currentState(): JSONObject {
        publishState()
        return latestState
    }

    private fun publishState() {
        handler.post {
            val root = rootInActiveWindow ?: return@post
            val newMap = HashMap<String, AccessibilityNodeInfo>()
            val elements = JSONArray()
            val packageName = root.packageName?.toString() ?: "unknown"
            val activity = root.className?.toString() ?: ""
            var counter = 0

            fun visit(node: AccessibilityNodeInfo, depth: Int, parentId: String?) {
                if (counter >= 400) return

                val rect = Rect()
                node.getBoundsInScreen(rect)

                val id = stableId(
                    packageName,
                    node.viewIdResourceName ?: "",
                    node.className?.toString() ?: "",
                    node.text?.toString() ?: "",
                    node.contentDescription?.toString() ?: "",
                    rect,
                    counter
                )

                newMap[id] = node

                val o = JSONObject()
                o.put("id", id)
                o.put("class_name", node.className?.toString() ?: "")
                o.put("text", node.text?.toString() ?: "")
                o.put("content_description", node.contentDescription?.toString() ?: "")
                o.put("resource_id", node.viewIdResourceName ?: "")
                o.put("package", node.packageName?.toString() ?: packageName)
                o.put("clickable", node.isClickable)
                o.put("long_clickable", node.isLongClickable)
                o.put("editable", node.isEditable)
                o.put("scrollable", node.isScrollable)
                o.put("enabled", node.isEnabled)
                o.put("focused", node.isFocused)
                o.put("selected", node.isSelected)
                o.put("checked", node.isChecked)
                o.put("visible", node.isVisibleToUser)
                val b = JSONObject()
                b.put("left", rect.left)
                b.put("top", rect.top)
                b.put("right", rect.right)
                b.put("bottom", rect.bottom)
                o.put("bounds", b)
                o.put("depth", depth)
                if (parentId != null) o.put("parent_id", parentId)

                elements.put(o)
                counter++

                for (i in 0 until node.childCount) {
                    node.getChild(i)?.let { child ->
                        visit(child, depth + 1, id)
                    }
                }
            }

            visit(root, 0, null)

            val state = JSONObject()
            state.put("revision", revision.incrementAndGet())
            state.put("package", packageName)
            state.put("activity", activity)
            state.put("screen", classify(packageName, activity))
            state.put("timestamp_ms", System.currentTimeMillis())
            state.put("elements", elements)

            val old = nodeMap
            nodeMap.clear()
            nodeMap.putAll(newMap)
            old.values.filter { it !== nodeMap.values }.forEach {
                try { it.recycle() } catch (_: Exception) {}
            }

            latestState = state
        }
    }

    private fun stableId(
        pkg: String,
        resource: String,
        clazz: String,
        text: String,
        desc: String,
        rect: Rect,
        counter: Int
    ): String {
        val raw = "$pkg|$resource|$clazz|$text|$desc|${rect.left},${rect.top},${rect.right},${rect.bottom}|$counter"
        return "e" + Integer.toUnsignedString(raw.hashCode(), 16)
    }

    private fun classify(pkg: String, activity: String): String {
        val p = pkg.lowercase()
        return when {
            p.contains("launcher") -> "HOME"
            p.contains("whatsapp") -> "WHATSAPP"
            p.contains("amazon") -> "AMAZON"
            p.contains("chrome") || p.contains("browser") -> "BROWSER"
            p.contains("dialer") || p.contains("incall") || p.contains("telecom") -> "CALL"
            else -> "APP"
        }
    }

    fun hierarchyXml(): String {
        val root = rootInActiveWindow ?: return "<hierarchy/>"
        fun esc(s: String): String =
            s.replace("&", "&amp;").replace("\"", "&quot;")
                .replace("<", "&lt;").replace(">", "&gt;")

        fun nodeXml(node: AccessibilityNodeInfo, depth: Int): String {
            val rect = Rect()
            node.getBoundsInScreen(rect)
            val indent = "  ".repeat(depth)
            val sb = StringBuilder()
            sb.append(indent).append("<node")
            sb.append(" class=\"").append(esc(node.className?.toString() ?: "")).append("\"")
            sb.append(" text=\"").append(esc(node.text?.toString() ?: "")).append("\"")
            sb.append(" content-desc=\"").append(esc(node.contentDescription?.toString() ?: "")).append("\"")
            sb.append(" resource-id=\"").append(esc(node.viewIdResourceName ?: "")).append("\"")
            sb.append(" clickable=\"").append(node.isClickable).append("\"")
            sb.append(" editable=\"").append(node.isEditable).append("\"")
            sb.append(" scrollable=\"").append(node.isScrollable).append("\"")
            sb.append(" bounds=\"").append(rect.toShortString()).append("\"")
            if (node.childCount == 0) {
                sb.append("/>\n")
            } else {
                sb.append(">\n")
                for (i in 0 until node.childCount) {
                    node.getChild(i)?.let { child ->
                        sb.append(nodeXml(child, depth + 1))
                    }
                }
                sb.append(indent).append("</node>\n")
            }
            return sb.toString()
        }
        return "<hierarchy>\n" + nodeXml(root, 1) + "</hierarchy>"
    }

    fun executeAction(action: JSONObject): JSONObject {
        val name = action.optString("action")
        return try {
            when (name) {
                "HOME" -> performGlobalAction(GLOBAL_ACTION_HOME)
                "BACK" -> performGlobalAction(GLOBAL_ACTION_BACK)
                "RECENTS" -> performGlobalAction(GLOBAL_ACTION_RECENTS)
                "WAIT" -> {
                    Thread.sleep((action.optDouble("seconds", 1.0) * 1000).toLong())
                    true
                }
                "OPEN_APP" -> {
                    val pkg = action.optString("app")
                    val intent = packageManager.getLaunchIntentForPackage(pkg)
                    if (intent != null) {
                        intent.addFlags(android.content.Intent.FLAG_ACTIVITY_NEW_TASK)
                        startActivity(intent)
                        true
                    } else false
                }
                "CLICK" -> click(action.optString("target_id"))
                "LONG_PRESS" -> longPress(action.optString("target_id"))
                "TYPE" -> typeText(action.optString("target_id"), action.optString("text"))
                "CLEAR" -> typeText(action.optString("target_id"), "")
                "SCROLL" -> scroll(action.optString("target_id"), action.optString("direction"))
                else -> false
            }.let { ok ->
                handler.postDelayed({ publishState() }, 350)
                JSONObject()
                    .put("ok", ok)
                    .put("message", if (ok) "executed $name" else "failed $name")
                    .put("revision", revision.get())
            }
        } catch (e: Exception) {
            JSONObject().put("ok", false).put("message", e.message ?: "action error")
                .put("revision", revision.get())
        }
    }

    private fun nodeFor(id: String): AccessibilityNodeInfo? {
        val n = nodeMap[id] ?: return null
        if (!n.isVisibleToUser || !n.isEnabled) return null
        return n
    }

    private fun click(id: String): Boolean {
        val n = nodeFor(id) ?: return false
        if (n.performAction(AccessibilityNodeInfo.ACTION_CLICK)) return true
        return tapCenter(n)
    }

    private fun longPress(id: String): Boolean {
        val n = nodeFor(id) ?: return false
        val r = Rect()
        n.getBoundsInScreen(r)
        val x = (r.left + r.right) / 2f
        val y = (r.top + r.bottom) / 2f
        val path = Path()
        path.moveTo(x, y)
        val stroke = GestureDescription.StrokeDescription(path, 0, 800)
        return dispatchGesture(
            GestureDescription.Builder().addStroke(stroke).build(),
            null,
            null
        )
    }

    private fun tapCenter(n: AccessibilityNodeInfo): Boolean {
        val r = Rect()
        n.getBoundsInScreen(r)
        val path = Path()
        path.moveTo((r.left + r.right) / 2f, (r.top + r.bottom) / 2f)
        return dispatchGesture(
            GestureDescription.Builder()
                .addStroke(GestureDescription.StrokeDescription(path, 0, 50))
                .build(), null, null
        )
    }

    private fun typeText(id: String, text: String): Boolean {
        val n = nodeFor(id) ?: return false
        if (!n.isEditable && !n.isFocused) return false
        val b = Bundle()
        b.putCharSequence(
            AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE,
            text
        )
        return n.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, b)
    }

    private fun scroll(id: String, direction: String): Boolean {
        val target = if (id.isNotBlank()) nodeFor(id) else
            nodeMap.values.firstOrNull { it.isScrollable && it.isVisibleToUser }

        target ?: return false
        val action = when (direction) {
            "up", "left" -> AccessibilityNodeInfo.ACTION_SCROLL_BACKWARD
            "down", "right" -> AccessibilityNodeInfo.ACTION_SCROLL_FORWARD
            else -> return false
        }
        return target.performAction(action)
    }
}
