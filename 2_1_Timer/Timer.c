#include "stm32f10x.h"                  // Device header

/**
  * 函    数：定时中断初始化
  * 参    数：无
  * 返 回 值：无
  */
void TIM2_Int_Init(u16 arr, u16 psc)
{
    TIM_TimeBaseInitTypeDef TIM_TimeBaseStructure;
    NVIC_InitTypeDef NVIC_InitStructure;

    // 使能 TIM2 时钟
    RCC_APB1PeriphClockCmd(RCC_APB1Periph_TIM2, ENABLE);

    // 定时器 TIM2 初始化
    TIM_TimeBaseStructure.TIM_Period = arr - 1; // 设置自动重装载寄存器周期的值
    TIM_TimeBaseStructure.TIM_Prescaler = psc - 1; // 设置预分频值
    TIM_TimeBaseStructure.TIM_ClockDivision = TIM_CKD_DIV1; // 设置时钟分割: TDTS = Tck_tim
    TIM_TimeBaseStructure.TIM_CounterMode = TIM_CounterMode_Up; // TIM 向上计数模式
    TIM_TimeBaseInit(TIM2, &TIM_TimeBaseStructure); // 初始化 TIM2 时间基数单位

    TIM_ITConfig(TIM2, TIM_IT_Update, ENABLE); // 使能指定的 TIM2 中断, 允许更新中断

    // 中断优先级 NVIC 设置
    NVIC_InitStructure.NVIC_IRQChannel = TIM2_IRQn; // TIM2 中断
    NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = 0; // 先占优先级 0 级
    NVIC_InitStructure.NVIC_IRQChannelSubPriority = 3; // 从优先级 3 级
    NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE; // IRQ 通道被使能
    NVIC_Init(&NVIC_InitStructure); // 初始化 NVIC 寄存器

    TIM_Cmd(TIM2, ENABLE); // 使能 TIM2
}


/* 定时器中断函数，可以复制到使用它的地方
void TIM2_IRQHandler(void)
{
	if (TIM_GetITStatus(TIM2, TIM_IT_Update) == SET)
	{
		
		TIM_ClearITPendingBit(TIM2, TIM_IT_Update);
	}
}
*/
