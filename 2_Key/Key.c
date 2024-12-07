#include "stm32f10x.h"
#include "key.h"
#include "sys.h" 
#include "delay.h"
#include "stdio.h"

/**************************************************************************************************** 
*                             长按、单击、双击定义
* 长按事件：任何大于 KEY_LONG_PRESS_TIME 
* 单击事件：按下时间不超过 KEY_LONG_PRESS_TIME 且 释放后 KEY_WAIT_DOUBLE_TIME 内无再次按下的操作
* 双击事件：俩次短按时间间隔小于KEY_WAIT_DOUBLE_TIME，俩次短按操作合并为一次双击事件。
* 特殊说明：
*          1.短按和长按时间间隔小于 KEY_WAIT_DOUBLE_TIME，响应一次单击和长按事件，不响应双击事件
*          2.连续2n次短按，且时间间隔小于 KEY_WAIT_DOUBLE_TIME，响应为n次双击
*          3.连续2n+1次短按，且时间间隔小于 KEY_WAIT_DOUBLE_TIME，且最后一次KEY_WAIT_DOUBLE_TIME内无操作，
*				响应为n次双击 和 一次单击事件
****************************************************************************************************/
#define KEY_LONG_PRESS_TIME    50 // 20ms*50 = 1s
#define KEY_WAIT_DOUBLE_TIME   25 // 20ms*25 = 500ms

#define KEY_PRESSED_LEVEL      0  // 按键按下是电平为低
/**************************************************************************************************** 
*                             局部函数定义
****************************************************************************************************/
static KEY_PinLevel_TypeDef KEY_ReadPin(u8 data);   // 按键读取按键的电平函数
static void KEY_GetAction_PressOrRelease(u8 data); // 获取按键是按下还是释放，保存到结构体

/**************************************************************************************************** 
*                             全局变量
****************************************************************************************************/
//KEY_Configure_TypeDef KeyCfg[data]={		
//		0,											//按键长按计数
//		KEY_Action_Release,			//虚拟当前IO电平，按下1，抬起0
//		KEY_Status_Idle,        //按键状态
//		KEY_Event_Null,         //按键事件
//		KEY_ReadPin             //读IO电平函数
//};

KEY_Configure_TypeDef KeyCfg[] = {
		{0, KEY_Action_Release, KEY_Status_Idle, KEY_Event_Null, KEY_ReadPin}, // 按键0
    {0, KEY_Action_Release, KEY_Status_Idle, KEY_Event_Null, KEY_ReadPin}, // 按键1
    {0, KEY_Action_Release, KEY_Status_Idle, KEY_Event_Null, KEY_ReadPin}, // 按键2
    {0, KEY_Action_Release, KEY_Status_Idle, KEY_Event_Null, KEY_ReadPin}, // 按键3
    {0, KEY_Action_Release, KEY_Status_Idle, KEY_Event_Null, KEY_ReadPin}, // 按键4
    {0, KEY_Action_Release, KEY_Status_Idle, KEY_Event_Null, KEY_ReadPin}, // 按键5
    {0, KEY_Action_Release, KEY_Status_Idle, KEY_Event_Null, KEY_ReadPin}, // 按键6
    {0, KEY_Action_Release, KEY_Status_Idle, KEY_Event_Null, KEY_ReadPin}, // 按键7
    {0, KEY_Action_Release, KEY_Status_Idle, KEY_Event_Null, KEY_ReadPin}, // 按键8
    {0, KEY_Action_Release, KEY_Status_Idle, KEY_Event_Null, KEY_ReadPin}, // 按键9
    {0, KEY_Action_Release, KEY_Status_Idle, KEY_Event_Null, KEY_ReadPin}, // 按键10
    {0, KEY_Action_Release, KEY_Status_Idle, KEY_Event_Null, KEY_ReadPin}, // 按键11
    {0, KEY_Action_Release, KEY_Status_Idle, KEY_Event_Null, KEY_ReadPin}, // 按键12
		{0, KEY_Action_Release, KEY_Status_Idle, KEY_Event_Null, KEY_ReadPin}, // 按键13
    {0, KEY_Action_Release, KEY_Status_Idle, KEY_Event_Null, KEY_ReadPin}, // 按键14
    {0, KEY_Action_Release, KEY_Status_Idle, KEY_Event_Null, KEY_ReadPin}  // 按键15
};

/**************************************************************************************************** 
*                             函数定义
****************************************************************************************************/
// 按键读取按键的电平函数，更具实际情况修改
#define KEY_GPIO GPIOC

static KEY_PinLevel_TypeDef KEY_ReadPin(u8 data) // 按键读取函数
{
    // 引脚映射数组
    static const uint16_t GPIO_Pins[] = {
        GPIO_Pin_1, GPIO_Pin_2, GPIO_Pin_3, GPIO_Pin_4,
        GPIO_Pin_5, GPIO_Pin_6, GPIO_Pin_7, GPIO_Pin_8,
        GPIO_Pin_9, GPIO_Pin_10, GPIO_Pin_11, GPIO_Pin_12,
        GPIO_Pin_13, GPIO_Pin_14,GPIO_Pin_15
    };

    // 输入范围校验
    if (data < 1 || data > 15) {
        return (KEY_PinLevel_TypeDef)-1; // 无效值
    }

    // 读取指定引脚的状态
    return (KEY_PinLevel_TypeDef)GPIO_ReadInputDataBit(KEY_GPIO, GPIO_Pins[data - 1]);
}


// 获取按键是按下还是释放，保存到结构体
static void KEY_GetAction_PressOrRelease(u8 data) // 根据实际按下按钮的电平去把它换算成虚拟的结果
{
	if(KeyCfg[data].KEY_ReadPin_Fcn(data) == KEY_PRESSED_LEVEL)
	{
		KeyCfg[data].KEY_Action = KEY_Action_Press;
	}
	else
	{
		KeyCfg[data].KEY_Action =  KEY_Action_Release;
	}
}

//按键初始化函数
void KEY_Init(void) //IO初始化
{ 
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOC,ENABLE);//使能PORTA时钟
	
 	GPIO_InitTypeDef GPIO_InitStructure;
 
	//初始化 WK_UP-->GPIOA.0	  
	GPIO_InitStructure.GPIO_Pin  = GPIO_Pin_13;
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IPU; //设置成上拉输入
	GPIO_Init(GPIOC, &GPIO_InitStructure);//初始化GPIOA.0
}

/**************************************************************************************************** 
*                             读取按键状态机
****************************************************************************************************/
void KEY_ReadStateMachine(u8 data)
{
	KEY_GetAction_PressOrRelease(data);
	
	switch(KeyCfg[data].KEY_Status)
	{
		//状态：没有按键按下
		case KEY_Status_Idle:
			if(KeyCfg[data].KEY_Action == KEY_Action_Press)
			{
				KeyCfg[data].KEY_Status = KEY_Status_Debounce;
				KeyCfg[data].KEY_Event = KEY_Event_Null;
			}
			else
			{
				KeyCfg[data].KEY_Status = KEY_Status_Idle;
				KeyCfg[data].KEY_Event = KEY_Event_Null;
			}
			break;
			
		//状态：消抖
		case KEY_Status_Debounce:
			if(KeyCfg[data].KEY_Action == KEY_Action_Press)
			{
				KeyCfg[data].KEY_Status = KEY_Status_ConfirmPress;
				KeyCfg[data].KEY_Event = KEY_Event_Null;
			}
			else
			{
				KeyCfg[data].KEY_Status = KEY_Status_Idle;
				KeyCfg[data].KEY_Event = KEY_Event_Null;
			}
			break;	


		//状态：继续按下
		case KEY_Status_ConfirmPress:
			if( (KeyCfg[data].KEY_Action == KEY_Action_Press) && ( KeyCfg[data].KEY_Count >= KEY_LONG_PRESS_TIME))
			{
				KeyCfg[data].KEY_Status = KEY_Status_ConfirmPressLong;
				KeyCfg[data].KEY_Event = KEY_Event_Null;
				KeyCfg[data].KEY_Count = 0;
			}
			else if( (KeyCfg[data].KEY_Action == KEY_Action_Press) && (KeyCfg[data].KEY_Count < KEY_LONG_PRESS_TIME))
			{
				KeyCfg[data].KEY_Count++;
				KeyCfg[data].KEY_Status = KEY_Status_ConfirmPress;
				KeyCfg[data].KEY_Event = KEY_Event_Null;
			}
			else
			{
				KeyCfg[data].KEY_Count = 0;
				KeyCfg[data].KEY_Status = KEY_Status_WaiteAgain;// 按短了后释放
				KeyCfg[data].KEY_Event = KEY_Event_Null;

			}
			break;	
			
		//状态：一直长按着
		case KEY_Status_ConfirmPressLong:
			if(KeyCfg[data].KEY_Action == KEY_Action_Press) 
			{   // 一直等待其放开
				KeyCfg[data].KEY_Status = KEY_Status_ConfirmPressLong;
				KeyCfg[data].KEY_Event = KEY_Event_Null;
				KeyCfg[data].KEY_Count = 0;
			}
			else
			{
				KeyCfg[data].KEY_Status = KEY_Status_Idle;
				KeyCfg[data].KEY_Event = KEY_Event_LongPress;
				KeyCfg[data].KEY_Count = 0;
			}
			break;	
			
		//状态：等待是否再次按下
		case KEY_Status_WaiteAgain:
			if((KeyCfg[data].KEY_Action != KEY_Action_Press) && ( KeyCfg[data].KEY_Count >= KEY_WAIT_DOUBLE_TIME))
			{   // 第一次短按,且释放时间大于KEY_WAIT_DOUBLE_TIME
				KeyCfg[data].KEY_Count = 0;
				KeyCfg[data].KEY_Status = KEY_Status_Idle;  
				KeyCfg[data].KEY_Event = KEY_Event_SingleClick;// 响应单击
				
			}
			else if((KeyCfg[data].KEY_Action != KEY_Action_Press) && ( KeyCfg[data].KEY_Count < KEY_WAIT_DOUBLE_TIME))
			{// 第一次短按,且释放时间还没到KEY_WAIT_DOUBLE_TIME
				KeyCfg[data].KEY_Count ++;
				KeyCfg[data].KEY_Status = KEY_Status_WaiteAgain;// 继续等待
				KeyCfg[data].KEY_Event = KEY_Event_Null;
				
			}
			else // 第一次短按,且还没到KEY_WAIT_DOUBLE_TIME 第二次被按下
			{
				KeyCfg[data].KEY_Count = 0;
				KeyCfg[data].KEY_Status = KEY_Status_SecondPress;// 第二次按下
				KeyCfg[data].KEY_Event = KEY_Event_Null;
			}
			break;		
		case KEY_Status_SecondPress:
			if( (KeyCfg[data].KEY_Action == KEY_Action_Press) && ( KeyCfg[data].KEY_Count >= KEY_LONG_PRESS_TIME))
			{
				KeyCfg[data].KEY_Status = KEY_Status_ConfirmPressLong;// 第二次按的时间大于 KEY_LONG_PRESS_TIME
				KeyCfg[data].KEY_Event = KEY_Event_SingleClick; // 先响应单击
				KeyCfg[data].KEY_Count = 0;
			}
			else if( (KeyCfg[data].KEY_Action == KEY_Action_Press) && ( KeyCfg[data].KEY_Count < KEY_LONG_PRESS_TIME))
			{
				KeyCfg[data].KEY_Count ++;
				KeyCfg[data].KEY_Status = KEY_Status_SecondPress;
				KeyCfg[data].KEY_Event = KEY_Event_Null;
			}
			else 
			{// 第二次按下后在 KEY_LONG_PRESS_TIME内释放
					KeyCfg[data].KEY_Count = 0;
					KeyCfg[data].KEY_Status = KEY_Status_Idle;
					KeyCfg[data].KEY_Event = KEY_Event_DoubleClick; // 响应双击
			}
			break;	
		default:
			break;
	}

}
