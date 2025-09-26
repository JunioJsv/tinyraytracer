#include <QtWidgets>

#include "mainwindow.h"

void onRender(const float &deltaTime) {
    qInfo() << "Frame rendered in" << deltaTime << "seconds";
}

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    MainWindow window;
    QObject::connect(&window, &MainWindow::onRender, onRender);
    window.show();
    return app.exec();
}