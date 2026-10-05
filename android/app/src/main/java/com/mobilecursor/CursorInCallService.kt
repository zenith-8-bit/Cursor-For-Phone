package com.mobilecursor

import android.telecom.Call
import android.telecom.InCallService

class CursorInCallService : InCallService() {

    private val callbacks = mutableMapOf<Call, Call.Callback>()

    override fun onCallAdded(call: Call) {
        super.onCallAdded(call)

        val callback = object : Call.Callback() {
            override fun onStateChanged(call: Call, state: Int) {
                // Future bridge event:
                // incoming/ringing/active/disconnected state can be forwarded
                // to the Python call controller.
            }
        }

        callbacks[call] = callback
        call.registerCallback(callback)

        // Intentionally not auto-answering here.
        // Android Telecom role/default-phone-app rules must be respected.
    }

    override fun onCallRemoved(call: Call) {
        callbacks.remove(call)?.let { call.unregisterCallback(it) }
        super.onCallRemoved(call)
    }

    fun answer(call: Call) {
        call.answer(0)
    }

    fun hangup(call: Call) {
        call.disconnect()
    }
}
