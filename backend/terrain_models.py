"""Application model choices shared by normal and supervised server launches."""
import threading

from flask import jsonify, request, send_from_directory


def register_model_routes(server):
    models = server.live_models()
    switching = threading.Event()

    @server.app.after_request
    def model_no_cache(response):
        if request.path in ('/api/models','/api/distill/models'):
            response.headers['Cache-Control']='no-store'
        return response

    @server.app.get('/terrain_model_controls.js')
    def model_controls():
        return send_from_directory(server.WEB_ROOT,'terrain_model_controls.js')

    @server.app.get('/api/models')
    @server.app.get('/api/distill/models')
    def model_choices():
        return jsonify(models.public() | dict(switching=switching.is_set()))

    @server.app.post('/api/models')
    @server.app.post('/api/distill/models')
    def change_models():
        with server.generation_coordinator.lock:
            if switching.is_set():
                return jsonify(error='A model change is already in progress.'),409
            restart = server.app.extensions.get('terrain_model_restart')
            if restart is None and server.RESTART_COMMAND is None:
                return jsonify(error='Restart this server through launch_terrain.py to change models.'),503
            try:
                config=models.save_request(request.get_json(silent=True))
            except ValueError as exc:
                return jsonify(error=str(exc)),400
            switching.set()
            server.generation_coordinator.suspend(True)
            if restart is not None:
                restart()
            else:
                server.server_restart()
            return jsonify(restarting=True,revision=config['revision']),202
