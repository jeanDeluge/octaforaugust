from flask import Flask
import config

# sql session 들어가야함

def create_app():
    app = Flask(__name__)
    app.config.from_object(config)

    # from . import models
    from .views import main_views, user_views, librarian_views
    app.register_blueprint(main_views.bp)
    app.register_blueprint(user_views.bp)
    app.register_blueprint(librarian_views.bp)


    return app