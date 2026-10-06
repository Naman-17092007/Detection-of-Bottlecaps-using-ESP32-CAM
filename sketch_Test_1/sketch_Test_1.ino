#include <Arduino.h>
#include <WiFi.h>
#include "esp_camera.h"
#include "esp_http_server.h"
#include "img_converters.h"

#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"

#include <Numsie-project-1_inferencing.h>
#include "edge-impulse-sdk/dsp/image/image.hpp"

// ==========================================
// Network Configuration (Set your local credentials)
// ==========================================
const char* ssid     = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// ==========================================
// AI-Thinker ESP32-CAM Pin Definitions
// ==========================================
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// ==========================================
// HTTP multipart constants
// ==========================================
#define PART_BOUNDARY "123456789000000000000987654321"
static const char* _STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* _STREAM_BOUNDARY     = "\r\n--" PART_BOUNDARY "\r\n";
static const char* _STREAM_PART         = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

httpd_handle_t stream_httpd = NULL;

// ==========================================
// Shared state
// ==========================================
static portMUX_TYPE sharedMux = portMUX_INITIALIZER_UNLOCKED;
static char  live_prediction_str[128] = "Initializing model...";
static float bottlecap_confidence     = 0.0f;

static uint8_t *snapshot_buf      = NULL;
static size_t   snapshot_buf_size = 0;

// ==========================================
// Channel detection
// ---------------------------------------------------------------
// NN_INPUT_FRAME_SIZE == W*H    -> grayscale (1 channel)
// NN_INPUT_FRAME_SIZE == W*H*3  -> RGB       (3 channels)
// ==========================================
#if (EI_CLASSIFIER_NN_INPUT_FRAME_SIZE == (EI_CLASSIFIER_INPUT_WIDTH * EI_CLASSIFIER_INPUT_HEIGHT))
    #define EI_IMG_IS_GRAYSCALE 1
#elif (EI_CLASSIFIER_NN_INPUT_FRAME_SIZE == (EI_CLASSIFIER_INPUT_WIDTH * EI_CLASSIFIER_INPUT_HEIGHT * 3))
    #define EI_IMG_IS_GRAYSCALE 0
#else
    // Fallback: assume grayscale if not sure
    #define EI_IMG_IS_GRAYSCALE 1
    #warning "EI_CLASSIFIER_NN_INPUT_FRAME_SIZE does not match W*H or W*H*3 -- defaulting to grayscale"
#endif

// ==========================================
// Feature callback
// ==========================================
static int raw_feature_get_data(size_t offset, size_t length, float *out_ptr) {
    size_t pixel_ix = offset * 3;   // snapshot_buf is always RGB888
#if EI_IMG_IS_GRAYSCALE
    for (size_t i = 0; i < length; i++) {
        uint32_t r = snapshot_buf[pixel_ix + 0];
        uint32_t g = snapshot_buf[pixel_ix + 1];
        uint32_t b = snapshot_buf[pixel_ix + 2];
        out_ptr[i] = (float)((r * 76 + g * 150 + b * 30) >> 8);   // BT.601
        pixel_ix += 3;
    }
#else
    for (size_t i = 0; i < length; i++) {
        uint32_t r = snapshot_buf[pixel_ix + 0];
        uint32_t g = snapshot_buf[pixel_ix + 1];
        uint32_t b = snapshot_buf[pixel_ix + 2];
        out_ptr[i] = (float)((r << 16) | (g << 8) | b);
        pixel_ix += 3;
    }
#endif
    return 0;
}

// ==========================================
// HTTP Handlers
// ==========================================
static esp_err_t index_handler(httpd_req_t *req) {
    const char* html_page =
    "<!DOCTYPE html><html><head><meta charset='utf-8'><title>ESP32-CAM Detector</title>"
    "<meta name='viewport' content='width=device-width, initial-scale=1'>"
    "<style>"
    "body{font-family:sans-serif;background:#181818;color:#eee;text-align:center;margin:0;padding:15px;}"
    "h2{margin:8px 0;font-size:22px;}"
    ".stream-box{max-width:480px;margin:0 auto;background:#222;padding:8px;border-radius:8px;box-shadow:0 4px 10px rgba(0,0,0,0.5);}"
    "img{width:100%;height:auto;border-radius:4px;display:block;}"
    ".card{max-width:480px;margin:15px auto;background:#282828;border-radius:8px;padding:12px;font-size:16px;text-align:left;}"
    ".badge{float:right;padding:3px 8px;border-radius:4px;background:#444;font-weight:bold;}"
    ".detected{background:#00c853 !important;color:#fff;}"
    "</style></head><body>"
    "<h2>AI-Thinker ESP32-CAM Live Feed</h2>"
    "<div class='stream-box'><img src='/stream'></div>"
    "<div class='card'>"
    "<div>Prediction: <span id='pred' style='font-weight:bold;'>Waiting...</span></div>"
    "<div style='margin-top:8px;'>Detection Alert: <span id='badge' class='badge'>NO TARGET</span></div>"
    "</div>"
    "<script>"
    "setInterval(function(){"
    "  fetch('/status').then(r=>r.json()).then(data=>{"
    "    document.getElementById('pred').innerText = data.prediction;"
    "    let badge = document.getElementById('badge');"
    "    if(data.detected){"
    "      badge.innerText = 'BOTTLECAP FOUND!';"
    "      badge.className = 'badge detected';"
    "    } else {"
    "      badge.innerText = 'CLEAR';"
    "      badge.className = 'badge';"
    "    }"
    "  }).catch(e=>{});"
    "}, 250);"
    "</script></body></html>";

    httpd_resp_set_type(req, "text/html");
    return httpd_resp_send(req, html_page, strlen(html_page));
}

static esp_err_t status_handler(httpd_req_t *req) {
    char json_buf[192];
    float local_conf;
    char  local_pred[128];

    portENTER_CRITICAL(&sharedMux);
    local_conf = bottlecap_confidence;
    strncpy(local_pred, live_prediction_str, sizeof(local_pred));
    local_pred[sizeof(local_pred) - 1] = '\0';
    portEXIT_CRITICAL(&sharedMux);

    snprintf(json_buf, sizeof(json_buf),
             "{\"prediction\":\"%s\",\"confidence\":%.2f,\"detected\":%s}",
             local_pred, local_conf,
             (local_conf > 0.75f) ? "true" : "false");

    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, json_buf, strlen(json_buf));
}

static esp_err_t stream_handler(httpd_req_t *req) {
    camera_fb_t *fb = NULL;
    esp_err_t res = ESP_OK;
    char part_buf[64];

    res = httpd_resp_set_type(req, _STREAM_CONTENT_TYPE);
    if (res != ESP_OK) return res;

    while (true) {
        fb = esp_camera_fb_get();
        if (!fb) {
            vTaskDelay(pdMS_TO_TICKS(10));
            continue;
        }

        uint8_t *jpg_buf = NULL;
        size_t   jpg_len = 0;

        bool converted = frame2jpg(fb, 80, &jpg_buf, &jpg_len);
        esp_camera_fb_return(fb);
        fb = NULL;

        if (!converted) {
            vTaskDelay(pdMS_TO_TICKS(10));
            continue;
        }

        if (res == ESP_OK) {
            res = httpd_resp_send_chunk(req, _STREAM_BOUNDARY, strlen(_STREAM_BOUNDARY));
        }
        if (res == ESP_OK) {
            size_t hlen = snprintf(part_buf, 64, _STREAM_PART, jpg_len);
            res = httpd_resp_send_chunk(req, part_buf, hlen);
        }
        if (res == ESP_OK) {
            res = httpd_resp_send_chunk(req, (const char *)jpg_buf, jpg_len);
        }

        free(jpg_buf);

        if (res != ESP_OK) break;
        vTaskDelay(pdMS_TO_TICKS(10));
    }
    return res;
}

void startCameraServer() {
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.server_port = 80;
    config.ctrl_port   = 32768;
    config.stack_size  = 8192;

    httpd_uri_t index_uri  = { "/",       HTTP_GET, index_handler,  NULL };
    httpd_uri_t stream_uri = { "/stream", HTTP_GET, stream_handler, NULL };
    httpd_uri_t status_uri = { "/status", HTTP_GET, status_handler, NULL };

    if (httpd_start(&stream_httpd, &config) == ESP_OK) {
        httpd_register_uri_handler(stream_httpd, &index_uri);
        httpd_register_uri_handler(stream_httpd, &stream_uri);
        httpd_register_uri_handler(stream_httpd, &status_uri);
    }
}

// ==========================================
// Inference Task (pinned to Core 1)
// ==========================================
void inference_task(void *pvParameters) {
    uint8_t *raw_rgb = (uint8_t*)ps_malloc(320 * 240 * 3);
    if (!raw_rgb) {
        Serial.println("ERR: raw_rgb PSRAM allocation failed!");
        vTaskDelete(NULL);
        return;
    }

    // -------- Print every relevant model macro so we can see what we're dealing with --------
    Serial.println();
    Serial.println("[META] ========== Edge Impulse Model Metadata ==========");
    Serial.printf("[META] INPUT_WIDTH            = %d\n", (int)EI_CLASSIFIER_INPUT_WIDTH);
    Serial.printf("[META] INPUT_HEIGHT           = %d\n", (int)EI_CLASSIFIER_INPUT_HEIGHT);
    Serial.printf("[META] INPUT_FRAMES           = %d\n", (int)EI_CLASSIFIER_INPUT_FRAMES);
    Serial.printf("[META] NN_INPUT_FRAME_SIZE    = %d\n", (int)EI_CLASSIFIER_NN_INPUT_FRAME_SIZE);
    Serial.printf("[META] RAW_SAMPLE_COUNT       = %d\n", (int)EI_CLASSIFIER_RAW_SAMPLE_COUNT);
    Serial.printf("[META] RAW_SAMPLES_PER_FRAME  = %d\n", (int)EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME);
    Serial.printf("[META] DSP_INPUT_FRAME_SIZE   = %d\n", (int)EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE);
    Serial.printf("[META] W*H                    = %d\n", (int)(EI_CLASSIFIER_INPUT_WIDTH * EI_CLASSIFIER_INPUT_HEIGHT));
    Serial.printf("[META] W*H*3                  = %d\n", (int)(EI_CLASSIFIER_INPUT_WIDTH * EI_CLASSIFIER_INPUT_HEIGHT * 3));
    Serial.printf("[META] IS_GRAYSCALE (decided) = %d\n", (int)EI_IMG_IS_GRAYSCALE);
    Serial.printf("[META] LABEL_COUNT            = %d\n", (int)EI_CLASSIFIER_LABEL_COUNT);
    for (int i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
        Serial.printf("[META] LABEL[%d]               = %s\n", i, ei_classifier_inferencing_categories[i]);
    }
    Serial.println("[META] ================================================");
    Serial.println();

    const size_t model_pixels = EI_CLASSIFIER_INPUT_WIDTH * EI_CLASSIFIER_INPUT_HEIGHT;

    uint32_t last_dbg   = 0;
    uint32_t iter_count = 0;

    while (1) {
        iter_count++;
        Serial.printf("[LOOP] iteration %u starting\n", (unsigned)iter_count);

        camera_fb_t *fb = esp_camera_fb_get();
        if (!fb) {
            Serial.println("[LOOP] esp_camera_fb_get() returned NULL, retrying...");
            vTaskDelay(pdMS_TO_TICKS(40));
            continue;
        }
        Serial.printf("[LOOP] got frame %ux%u, %u bytes\n", (unsigned)fb->width, (unsigned)fb->height, (unsigned)fb->len);

        // RGB565 (big-endian) -> RGB888
        size_t   pixel_count = fb->width * fb->height;
        uint8_t *src = fb->buf;
        uint8_t *dst = raw_rgb;

        for (size_t i = 0; i < pixel_count; i++) {
            uint16_t pixel = ((uint16_t)src[0] << 8) | src[1];
            src += 2;

            *dst++ = (uint8_t)(((pixel >> 11) & 0x1F) * 255 / 31);
            *dst++ = (uint8_t)(((pixel >> 5)  & 0x3F) * 255 / 63);
            *dst++ = (uint8_t)(( pixel        & 0x1F) * 255 / 31);
        }

        esp_camera_fb_return(fb);
        fb = NULL;

        // Crop & scale
        ei::image::processing::crop_and_interpolate_rgb888(
            raw_rgb, 320, 240,
            snapshot_buf, EI_CLASSIFIER_INPUT_WIDTH, EI_CLASSIFIER_INPUT_HEIGHT);

        // Debug pixels every 2 s
        if (millis() - last_dbg > 2000) {
            last_dbg = millis();
            Serial.printf("[DBG] RGB[0..3]=%02X%02X%02X %02X%02X%02X %02X%02X%02X %02X%02X%02X\n",
                          snapshot_buf[0], snapshot_buf[1], snapshot_buf[2],
                          snapshot_buf[3], snapshot_buf[4], snapshot_buf[5],
                          snapshot_buf[6], snapshot_buf[7], snapshot_buf[8],
                          snapshot_buf[9], snapshot_buf[10], snapshot_buf[11]);
        }

        // Build signal
        signal_t signal;
        signal.total_length = model_pixels;   // one float per pixel (W*H)
        signal.get_data     = &raw_feature_get_data;

        Serial.printf("[LOOP] calling run_classifier (total_length=%u)...\n", (unsigned)signal.total_length);
        uint32_t t_start = millis();

        ei_impulse_result_t result = { 0 };
        EI_IMPULSE_ERROR res = run_classifier(&signal, &result, false);

        uint32_t t_elapsed = millis() - t_start;
        Serial.printf("[LOOP] run_classifier returned %d after %u ms\n", (int)res, (unsigned)t_elapsed);

        if (res == EI_IMPULSE_OK) {
            float local_cap_val  = 0.0f;
            float local_none_val = 0.0f;

            for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
                const char *lbl = result.classification[ix].label;
                Serial.printf("[LOOP]   class[%u] %-12s = %.4f\n",
                              (unsigned)ix, lbl, result.classification[ix].value);
                if (strcasecmp(lbl, "BOTTLECAP") == 0) {
                    local_cap_val = result.classification[ix].value;
                } else if (strcasecmp(lbl, "NONE") == 0 ||
                           strcasecmp(lbl, "BACKGROUND") == 0) {
                    local_none_val = result.classification[ix].value;
                }
            }

            char local_summary[128];
            snprintf(local_summary, sizeof(local_summary),
                     "Cap: %.1f%% | None: %.1f%% (%d ms)",
                     local_cap_val * 100.0f,
                     local_none_val * 100.0f,
                     result.timing.classification);

            portENTER_CRITICAL(&sharedMux);
            strncpy(live_prediction_str, local_summary, sizeof(live_prediction_str));
            live_prediction_str[sizeof(live_prediction_str) - 1] = '\0';
            bottlecap_confidence = local_cap_val;
            portEXIT_CRITICAL(&sharedMux);

            Serial.printf("[INFERENCE] %s\n", local_summary);
        } else {
            Serial.printf("[INFERENCE] run_classifier error %d\n", (int)res);
        }

        vTaskDelay(pdMS_TO_TICKS(100));
    }

    free(raw_rgb);
}

// ==========================================
// Setup & Loop
// ==========================================
void setup() {
    WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);

    Serial.begin(115200);
    uint32_t t0 = millis();
    while (!Serial && (millis() - t0) < 2000) { delay(10); }

    Serial.println("\n==========================================");
    Serial.println("  AI-Thinker ESP32-CAM Edge Server Boot   ");
    Serial.println("==========================================");

    if (!psramFound()) {
        Serial.println("FATAL: External PSRAM not found. Check Tools menu!");
        while (1) { delay(1000); }
    }
    Serial.printf("PSRAM Active: Total %d KB, Free %d KB\n",
                  ESP.getPsramSize() / 1024, ESP.getFreePsram() / 1024);

    pinMode(PWDN_GPIO_NUM, OUTPUT);
    digitalWrite(PWDN_GPIO_NUM, HIGH);
    delay(150);
    digitalWrite(PWDN_GPIO_NUM, LOW);
    delay(150);

    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer   = LEDC_TIMER_0;
    config.pin_d0       = Y2_GPIO_NUM;
    config.pin_d1       = Y3_GPIO_NUM;
    config.pin_d2       = Y4_GPIO_NUM;
    config.pin_d3       = Y5_GPIO_NUM;
    config.pin_d4       = Y6_GPIO_NUM;
    config.pin_d5       = Y7_GPIO_NUM;
    config.pin_d6       = Y8_GPIO_NUM;
    config.pin_d7       = Y9_GPIO_NUM;
    config.pin_xclk     = XCLK_GPIO_NUM;
    config.pin_pclk     = PCLK_GPIO_NUM;
    config.pin_vsync    = VSYNC_GPIO_NUM;
    config.pin_href     = HREF_GPIO_NUM;
    config.pin_sccb_sda = SIOD_GPIO_NUM;
    config.pin_sccb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn     = PWDN_GPIO_NUM;
    config.pin_reset    = RESET_GPIO_NUM;
    config.xclk_freq_hz = 10000000;
    config.pixel_format = PIXFORMAT_RGB565;
    config.frame_size   = FRAMESIZE_QVGA;
    config.jpeg_quality = 12;
    config.fb_count     = 2;
    config.grab_mode    = CAMERA_GRAB_LATEST;

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("FATAL: Camera init failed: 0x%x\n", err);
        while (1) { delay(1000); }
    }
    Serial.println("Camera initialized successfully.");

    sensor_t *s = esp_camera_sensor_get();
    if (s) {
        s->set_gain_ctrl(s, 0);
        s->set_exposure_ctrl(s, 0);
        s->set_whitebal(s, 0);
        s->set_awb_gain(s, 0);
        s->set_agc_gain(s, 0);
        s->set_aec_value(s, 300);
        s->set_brightness(s, 0);
        s->set_contrast(s, 0);
        s->set_saturation(s, 0);
    }

    snapshot_buf_size = EI_CLASSIFIER_INPUT_WIDTH * EI_CLASSIFIER_INPUT_HEIGHT * 3;
    snapshot_buf = (uint8_t*)ps_malloc(snapshot_buf_size);
    if (!snapshot_buf) {
        Serial.println("FATAL: Unable to allocate snapshot_buf in PSRAM");
        while (1) { delay(1000); }
    }
    Serial.printf("Model input buffer: %u bytes (RGB888 %dx%d)\n",
                  (unsigned)snapshot_buf_size,
                  EI_CLASSIFIER_INPUT_WIDTH, EI_CLASSIFIER_INPUT_HEIGHT);

    Serial.printf("Connecting to Wi-Fi: %s\n", ssid);
    WiFi.begin(ssid, password);
    int retry = 0;
    while (WiFi.status() != WL_CONNECTED && retry < 30) {
        delay(500);
        Serial.print(".");
        retry++;
    }

    if (WiFi.status() == WL_CONNECTED) {
        Serial.println("\nWi-Fi connected successfully!");
        Serial.print("Open web browser at: http://");
        Serial.println(WiFi.localIP());
        startCameraServer();
    } else {
        Serial.println("\nERR: Wi-Fi connection timed out.");
    }

    xTaskCreatePinnedToCore(
        inference_task,
        "InferenceTask",
        1024 * 16,
        NULL,
        1,
        NULL,
        1
    );
}

void loop() {
    vTaskDelay(pdMS_TO_TICKS(1000));
}