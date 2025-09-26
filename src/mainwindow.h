#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTimer>
#include "cuda/cuda_raytracer.h"

using namespace CudaRaytracer;

namespace Ui {
    class MainWindow;
}

class MainWindow final : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);

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

    float last_frame_time = 0.0F;
    QPoint window_center;
    float delta_time = 0.0F;
    vec3 position{0, 0, 0};
    float camera_yaw = 0;
    float camera_pitch = 0;
    int width;
    int height;
    float fov = 1.05;
};

#endif // MAINWINDOW_H
