#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTimer>
#include "raytracer.h"

using namespace Raytracer;

namespace Ui {
    class MainWindow;
}

class MainWindow final : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);

    static vec3 rotateCamera(vec3 v, float yaw, float pitch);

    QImage render() const;

    ~MainWindow() override;

signals:
    void onRender(const float &deltaTime);

protected:
    void keyPressEvent(QKeyEvent *event) override;

    void mouseMoveEvent(QMouseEvent *event) override;

private:
    Ui::MainWindow *ui;
    QTimer timer;

    float lastFrameTime = 0.0F;
    vec3 position{0, 0, 0};
    float camera_yaw = 0;
    float camera_pitch = 0;
    int width;
    int height;
    float fov = 1.05;
};

#endif // MAINWINDOW_H
