#include "control.h"
#include "stepping.h"

#define stepping_pul(x)			GPIO_WriteBit(GPIOA, GPIO_Pin_1, (BitAction)(x))
#define stepping_dir(x)		  GPIO_WriteBit(GPIOB, GPIO_Pin_8, (BitAction)(x))
#define stepping_ena(x)			GPIO_WriteBit(GPIOB, GPIO_Pin_0, (BitAction)(x))

/**
  * @brief  ��ת�Ǻ��жϽų�ʼ��
  * @param  ��
  * @retval ��
  */
void stepping_GPIO()
{
	
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOB, ENABLE);		//����GPIOA��ʱ��

	/*GPIO��ʼ��*/
	GPIO_InitTypeDef GPIO_InitStructure;
	
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_0 | GPIO_Pin_8;
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
	GPIO_Init(GPIOB, &GPIO_InitStructure);						

	GPIO_SetBits(GPIOB, GPIO_Pin_8);						 	//PB8�����  ˳ʱ�뷽��  DRIVER_DIR
	GPIO_ResetBits(GPIOB, GPIO_Pin_0);						//PA11����� ʹ�����    DRIVER_OE
}

/**
  * @brief  ʹ������ų�ʼ��
  * @param  ��
  * @retval ��
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
  * @brief  �ܳ�ʼ��
  * @param  ��
  * @retval ��
  */
void stepping_init(void)
{
	stepping_GPIO();
	stepping_GPIO_PWM();
}
/**
  * @brief  ���PWM
	* @param :angle �� �������������		
						speed :  ���������ת�ٶȣ�ʵ���Ͼ����޸���PWM��Ƶ�ʣ�
  * @retval ��
  */
void stepping_PWM(uint32_t angle,uint32_t speed)
{
	for(int i= 0; i < angle; i++)   //ģ��PWM���Ʋ������
	{
			stepping_pul(1);
			Delay_us(speed);  //�����޸���ʱ��������Ĳ��������Ĳ��������ת���ٶȡ���������ģ��ı���PWM��Ƶ�ʣ������ı����ٶ�
			stepping_pul(0);
			Delay_us(speed);
	}

}
/**
  * @brief  ����ת
  * @param  ��
  * @retval ��
  */
void stepping_reversal(void)
{
	static _Bool stepping_reversal_i;
	if(stepping_reversal_i)
			 {stepping_dir(0);stepping_reversal_i = 0;}
	else {stepping_dir(1);stepping_reversal_i = 1;}
}

/**
  * @brief  �жϷ�ת
  * @param  ��
  * @retval ��
  */
void stepping_End(void)
{
	static _Bool stepping_End_i;
	if(stepping_End_i)
			{stepping_ena(1);stepping_End_i = 0;}
	else {stepping_ena(0);stepping_End_i = 1;}
}
