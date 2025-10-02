#include <driver/ledc.h>
#include <esp_log.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <stdio.h>

static const char * TAG = "MAIN";

void heartbeat_task(void * pvParam)
{
    ledc_timer_config_t timer_config = {.speed_mode = LEDC_LOW_SPEED_MODE,
                                        .duty_resolution = LEDC_TIMER_10_BIT,
                                        .timer_num = LEDC_TIMER_0,
                                        .freq_hz = 1,
                                        .clk_cfg = LEDC_AUTO_CLK,
                                        .deconfigure = false};

    ledc_timer_config(&timer_config);
    ledc_channel_config_t channel_config = {.gpio_num = GPIO_NUM_2,
                                            .speed_mode = LEDC_LOW_SPEED_MODE,
                                            .channel = LEDC_CHANNEL_0,
                                            .intr_type = LEDC_INTR_DISABLE,
                                            .timer_sel = LEDC_TIMER_0,
                                            .duty = 1UL << (timer_config.duty_resolution - 1),
                                            .hpoint = 0,
                                            .sleep_mode = LEDC_SLEEP_MODE_KEEP_ALIVE,
                                            .flags = {.output_invert = 0}};

    ledc_channel_config(&channel_config);
    vTaskDelete(nullptr);
}

extern "C" void app_main(void)
{
    xTaskCreate(heartbeat_task, "LED Blink", configMINIMAL_STACK_SIZE * 2, nullptr, 5, nullptr);

    uint32_t count = 0;
    while (true)
    {
        ESP_LOGI(TAG, "Hello World! Count: %u", count++);
        vTaskDelay(pdMS_TO_TICKS(100));
    }
}
