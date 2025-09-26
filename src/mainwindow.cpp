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
    window_center = QPoint(width / 2, height / 2);
    setCursor(Qt::BlankCursor);
    connect(&timer, &QTimer::timeout, this, [=] {
        float current_frame_time = get_current_time_in_mills() / 1000.0F;
        delta_time = current_frame_time - last_frame_time;
        last_frame_time = current_frame_time;
        emit onRender(delta_time);
        QImage img = render();
        const QPixmap pixmap = QPixmap::fromImage(img);
        ui->frame->setPixmap(pixmap);
    });
    timer.start(16);
}

QImage MainWindow::render() const {
    QImage img(width, height, QImage::Format_RGB32);
    render_cuda(reinterpret_cast<uint32_t *>(img.bits()), width, height, position, camera_pitch, camera_yaw);
    return img;
}

MainWindow::~MainWindow() {
    delete ui;
}

void MainWindow::keyPressEvent(QKeyEvent *event) {
    constexpr float speed = 10.0f;
    switch (event->key()) {
        case Qt::Key_W: position += (vec3{0, 0, -1}).rotated(camera_pitch, camera_yaw) * speed * delta_time;
            break;
        case Qt::Key_S: position += (vec3{0, 0, 1}).rotated(camera_pitch, camera_yaw) * speed * delta_time;
            break;
        case Qt::Key_A: position += (vec3{-1, 0, 0}).rotated(camera_pitch, camera_yaw) * speed * delta_time;
            break;
        case Qt::Key_D: position += (vec3{1, 0, 0}).rotated(camera_pitch, camera_yaw) * speed * delta_time;
            break;
        default: QMainWindow::keyPressEvent(event);;
    }
}

void MainWindow::mouseMoveEvent(QMouseEvent *event) {
    constexpr float speed = 0.1f;

    if (event->buttons() & Qt::LeftButton) {
        QPoint delta = event->pos() - window_center;

        camera_yaw -= delta.x() * speed * delta_time;
        camera_pitch -= delta.y() * speed * delta_time;

        camera_pitch = qBound(-89.0f, camera_pitch, 89.0f);

        QPoint global_center = mapToGlobal(window_center);
        QCursor::setPos(global_center);
    }

    QMainWindow::mouseMoveEvent(event);
}
