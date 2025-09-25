#include "mainwindow.h"

#include <QtWidgets>

#include "ui_mainwindow.h"
#include "utils.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
      , ui(new Ui::MainWindow)
      , timer(this) {
    ui->setupUi(this);
    width = QMainWindow::width();
    height = QMainWindow::height();
    connect(&timer, &QTimer::timeout, this, [=] {
        float currentFrameTime = getCurrentTimeInMills() / 1000.0F;
        float deltaTime = currentFrameTime - lastFrameTime;
        lastFrameTime = currentFrameTime;
        emit onRender(deltaTime);
        QImage img = render();
        const QPixmap pixmap = QPixmap::fromImage(img);
        ui->frame->setPixmap(pixmap);
    });
    timer.start(33);
}

vec3 MainWindow::rotateCamera(vec3 v, float yaw, float pitch) {
    // pitch
    float cos_pitch = std::cos(pitch);
    float sin_pitch = std::sin(pitch);
    vec3 v_pitch = {
        v.x,
        v.y * cos_pitch - v.z * sin_pitch,
        v.y * sin_pitch + v.z * cos_pitch
    };

    // yaw
    float cos_yaw = std::cos(yaw);
    float sin_yaw = std::sin(yaw);
    return {
        v_pitch.x * cos_yaw + v_pitch.z * sin_yaw,
        v_pitch.y,
        -v_pitch.x * sin_yaw + v_pitch.z * cos_yaw
    };
}

QImage MainWindow::render() const {
    QImage img(width, height, QImage::Format_RGB32);
#pragma omp parallel for
    for (int pix = 0; pix < width * height; pix++) {
        // actual rendering loop
        float dir_x = (pix % width + 0.5) - width / 2.;
        float dir_y = -(pix / width + 0.5) + height / 2.; // this flips the image at the same time
        float dir_z = -height / (2. * tan(fov / 2.));
        vec3 dir = vec3{dir_x, dir_y, dir_z}.normalized();
        dir = rotateCamera(dir, camera_yaw, camera_pitch).normalized();
        vec3 color = cast_ray(position, dir);
        float max = std::max(1.f, std::max(color[0], std::max(color[1], color[2])));
        float r = color[0] / max, g = color[1] / max, b = color[2] / max;
        img.setPixel(
            pix % width, pix / width,
            qRgb(255 * r, 255 * g, 255 * b)
        );
    }
    return img;
}

MainWindow::~MainWindow() {
    delete ui;
}

void MainWindow::keyPressEvent(QKeyEvent *event) {
    switch (event->key()) {
        case Qt::Key_W: position = position + rotateCamera({0, 0, -1}, camera_yaw, camera_pitch) * 0.5;
            break;
        case Qt::Key_S: position = position + rotateCamera({0, 0, 1}, camera_yaw, camera_pitch) * 0.5;
            break;
        case Qt::Key_A: position = position + rotateCamera({-1, 0, 0}, camera_yaw, camera_pitch) * 0.5;
            break;
        case Qt::Key_D: position = position + rotateCamera({1, 0, 0}, camera_yaw, camera_pitch) * 0.5;
            break;
        default: QMainWindow::keyPressEvent(event);;
    }
}

void MainWindow::mouseMoveEvent(QMouseEvent *event) {
    static QPoint last_pos = event->pos();
    QPoint delta = event->pos() - last_pos;
    last_pos = event->pos();
    if (event->buttons() & Qt::LeftButton) {
        camera_yaw -= delta.x() * 0.005;
        camera_pitch -= delta.y() * 0.005;
    }
    QMainWindow::mouseMoveEvent(event);
}
