#include "control.h"
#include "stepping.h"

#define stepping_pul(x)			GPIO_WriteBit(GPIOA, GPIO_Pin_1, (BitAction)(x))
#define stepping_dir(x)		  GPIO_WriteBit(GPIOB, GPIO_Pin_8, (BitAction)(x))
#define stepping_ena(x)			GPIO_WriteBit(GPIOB, GPIO_Pin_0, (BitAction)(x))

/**
  * @brief  旋转角和中断脚初始化
  * @param  无
  * @retval 无
  */
void stepping_GPIO()
{
	
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOB, ENABLE);		//开启GPIOA的时钟

	/*GPIO初始化*/
	GPIO_InitTypeDef GPIO_InitStructure;
	
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_0 | GPIO_Pin_8;
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
	GPIO_Init(GPIOB, &GPIO_InitStructure);						

	GPIO_SetBits(GPIOB, GPIO_Pin_8);						 	//PB8输出高  顺时针方向  DRIVER_DIR
	GPIO_ResetBits(GPIOB, GPIO_Pin_0);						//PA11输出低 使能输出    DRIVER_OE
}

/**
  * @brief  使能脉冲脚初始化
  * @param  无
  * @retval 无
  */
void stepping_GPIO_PWM()
{	
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOA, ENABLE);
	GPIO_InitTypeDef GPIO_InitStructure;
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP; 
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_1; 
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
	GPIO_Init(GPIOA, &GPIO_InitStructure);
}
/**
  * @brief  总初始化
  * @param  无
  * @retval 无
  */
void stepping_init(void)
{
	stepping_GPIO();
	stepping_GPIO_PWM();
}
/**
  * @brief  软件PWM
	* @param :angle ： 步进电机脉冲数		
						speed :  步进电机旋转速度（实际上就是修改了PWM的频率）
  * @retval 无
  */
void stepping_PWM(uint32_t angle,uint32_t speed)
{
	for(int i= 0; i < angle; i++)   //模拟PWM控制步进电机
	{
			stepping_pul(1);
			Delay_us(speed);  //可以修改延时函数里面的参数，更改步进电机旋转的速度。本质上是模拟改变了PWM的频率，进而改变了速度
			stepping_pul(0);
			Delay_us(speed);
	}

}
/**
  * @brief  方向反转
  * @param  无
  * @retval 无
  */
void stepping_reversal(void)
{
	static _Bool stepping_reversal_i;
	if(stepping_reversal_i)
			 {stepping_dir(0);stepping_reversal_i = 0;}
	else {stepping_dir(1);stepping_reversal_i = 1;}
}

/**
  * @brief  中断反转
  * @param  无
  * @retval 无
  */
void stepping_End(void)
{
	static _Bool stepping_End_i;
	if(stepping_End_i)
			{stepping_ena(1);stepping_End_i = 0;}
	else {stepping_ena(0);stepping_End_i = 1;}
}
